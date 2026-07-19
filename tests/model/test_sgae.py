import pytest
import torch
from torch_geometric.data import Data, Batch

from model.sgae import SparseVariationalGraphAutoEncoder
from model.loss import kl_loss, elbo_loss


def _make_batch(num_channels: int, generator: torch.Generator, sizes: tuple[int, ...] = (5, 8)) -> Batch:
  data_list = []
  for num_nodes in sizes:
    x = torch.randn(num_nodes, num_channels, generator=generator)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), generator=generator)
    data_list.append(Data(x=x, edge_index=edge_index))
  return Batch.from_data_list(data_list)


# --- basic shape / construction -------------------------------------------

@pytest.mark.parametrize("conv_layer_type", ["GCNConv", "SAGEConv"])
@pytest.mark.parametrize("is_variational", [False, True])
def test_forward_shape(conv_layer_type: str, is_variational: bool) -> None:
  generator = torch.Generator().manual_seed(0)
  num_channels = 2
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=6,
    activation_fn="relu",
    conv_layer_type=conv_layer_type,
    num_layers=2,
    is_variational=is_variational,
  )

  x_hat, mask, mu, logstd = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                  sparsity_ratio=0.5, temperature=0.5)

  assert x_hat.shape == batch.x.shape
  assert mask.shape == (batch.num_nodes, 6)
  if is_variational:
    assert mu.shape == (batch.num_nodes, 6)
    assert logstd.shape == (batch.num_nodes, 6)
  else:
    assert mu is None and logstd is None


def test_single_layer_stack_forward_shape() -> None:
  generator = torch.Generator().manual_seed(7)
  num_channels = 3
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=5,
    activation_fn="relu",
    conv_layer_type="GCNConv",
    num_layers=1,
    is_variational=False,
  )

  x_hat, mask, mu, logstd = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                  sparsity_ratio=0.5, temperature=0.5)
  assert x_hat.shape == batch.x.shape
  assert mask.shape == (batch.num_nodes, 5)
  assert mu is None and logstd is None


@pytest.mark.parametrize("sparsity_ratio", [0.0, 1.0, -0.1, 1.5])
def test_invalid_sparsity_ratio_raises(sparsity_ratio: float) -> None:
  generator = torch.Generator().manual_seed(0)
  num_channels = 2
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  with pytest.raises(AssertionError):
    model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
         sparsity_ratio=sparsity_ratio, temperature=0.5)


# --- sparsification: per-graph budget + batch isolation --------------------

def test_sparsity_respects_per_graph_budget() -> None:
  generator = torch.Generator().manual_seed(1)
  num_channels = 2
  batch = _make_batch(num_channels, generator)
  sparsity_ratio = 0.5

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=6,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )
  model.eval()

  z, _, _, _ = model.encode(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                            sparsity_ratio=sparsity_ratio, temperature=0.1)

  for graph_id in batch.batch.unique().tolist():
    node_mask = batch.batch == graph_id
    num_entries = int(node_mask.sum()) * z.shape[-1]
    expected_k = max(1, round(sparsity_ratio * num_entries))
    actual_k = (z[node_mask] != 0).sum().item()
    assert actual_k == expected_k


def test_sparsification_batches_are_independent() -> None:
  generator = torch.Generator().manual_seed(6)
  num_channels = 2
  batch = _make_batch(num_channels, generator, sizes=(3, 4, 5))
  sparsity_ratio = 0.4

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  z = torch.randn(batch.num_nodes, 4, generator=generator)
  baseline, _ = model._ste_sparsify(z, batch.batch, sparsity_ratio=sparsity_ratio, temperature=0.1)

  for graph_id in batch.batch.unique().tolist():
    node_mask = batch.batch == graph_id
    num_entries = int(node_mask.sum()) * z.shape[-1]
    expected_k = max(1, round(sparsity_ratio * num_entries))
    assert (baseline[node_mask] != 0).sum().item() == expected_k

  # Blowing up graph 0's values must not change which entries graphs 1/2 select.
  z_perturbed = z.clone()
  z_perturbed[batch.batch == 0] *= 100
  perturbed, _ = model._ste_sparsify(z_perturbed, batch.batch, sparsity_ratio=sparsity_ratio, temperature=0.1)

  other_graphs = batch.batch != 0
  assert torch.equal(baseline[other_graphs] != 0, perturbed[other_graphs] != 0)


# --- STE: forward is the hard mask, backward flows through the soft mask ---

def test_ste_forward_matches_hard_topk_and_is_temperature_invariant() -> None:
  generator = torch.Generator().manual_seed(4)
  num_channels = 2
  batch = _make_batch(num_channels, generator, sizes=(3, 4))
  sparsity_ratio = 0.4

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=5,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  z = torch.randn(batch.num_nodes, 5, generator=generator)

  low_temp, _ = model._ste_sparsify(z, batch.batch, sparsity_ratio=sparsity_ratio, temperature=0.05)
  high_temp, _ = model._ste_sparsify(z, batch.batch, sparsity_ratio=sparsity_ratio, temperature=5.0)

  # The forward value is exactly the hard mask, regardless of temperature.
  assert torch.equal(low_temp, high_temp)

  for graph_id in batch.batch.unique().tolist():
    node_mask = batch.batch == graph_id
    block = z[node_mask]
    out_block = low_temp[node_mask]
    selected = out_block != 0

    # Selected entries keep their original value; unselected entries are zeroed.
    assert torch.equal(out_block[selected], block[selected])
    assert torch.all(out_block[~selected] == 0)

    # Selection is a per-channel magnitude threshold (budget is allocated per
    # channel, not flattened across the whole block): every selected entry's
    # magnitude is >= every unselected entry's magnitude within that channel.
    for c in range(block.shape[-1]):
      selected_mag = block[:, c].abs()[selected[:, c]]
      unselected_mag = block[:, c].abs()[~selected[:, c]]
      if selected_mag.numel() and unselected_mag.numel():
        assert selected_mag.min() >= unselected_mag.max()

    expected_k = max(1, round(sparsity_ratio * block.numel()))
    assert int(selected.sum()) == expected_k


def test_ste_gradient_reaches_unselected_entries() -> None:
  generator = torch.Generator().manual_seed(2)
  num_channels = 2
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=6,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  h = model._run_gnn(batch.x, batch.edge_index, model.encoder_layers, model.encoder_norms)
  h = h.detach().requires_grad_(True)
  z_sparse, _ = model._ste_sparsify(h, batch.batch, sparsity_ratio=0.3, temperature=0.5)
  z_sparse.sum().backward()

  unselected_grad = h.grad[z_sparse == 0]
  assert unselected_grad.numel() > 0
  assert (unselected_grad != 0).any()


def test_ste_gradient_sharpens_as_temperature_anneals() -> None:
  # Low temperature (annealed toward 0) should saturate the soft mask, making
  # gradients on off-threshold entries vanish; a higher/fixed temperature
  # keeps the soft mask smoother and those gradients larger.
  generator = torch.Generator().manual_seed(5)
  num_channels = 2
  batch = _make_batch(num_channels, generator, sizes=(6,))
  sparsity_ratio = 0.5

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  z = torch.randn(batch.num_nodes, 4, generator=generator)
  z[0, 0] = 0.01  # far below the selection threshold
  z = z.detach().requires_grad_(True)

  def grad_at_far_entry(temperature: float) -> float:
    if z.grad is not None:
      z.grad = None
    out, _ = model._ste_sparsify(z, batch.batch, sparsity_ratio=sparsity_ratio, temperature=temperature)
    out.sum().backward()
    return z.grad[0, 0].abs().item()

  grad_annealed = grad_at_far_entry(temperature=0.01)
  grad_fixed = grad_at_far_entry(temperature=2.0)

  assert grad_annealed < grad_fixed


def test_temperature_schedule_keeps_sparsity_exact_fixed_and_annealed() -> None:
  # The model holds no temperature state, so both a constant ("STE-fixed")
  # schedule and a decaying ("STE-annealing") schedule must be usable from
  # the caller, and the per-graph budget must hold exactly under either.
  generator = torch.Generator().manual_seed(9)
  num_channels = 2
  batch = _make_batch(num_channels, generator, sizes=(4, 7))
  sparsity_ratio = 0.4

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=False,
  )

  z = torch.randn(batch.num_nodes, 4, generator=generator)
  fixed_schedule = [0.5, 0.5, 0.5, 0.5]
  annealing_schedule = [1.0, 0.5, 0.1, 0.01]

  for temperature in fixed_schedule + annealing_schedule:
    out, _ = model._ste_sparsify(z, batch.batch, sparsity_ratio=sparsity_ratio, temperature=temperature)
    for graph_id in batch.batch.unique().tolist():
      node_mask = batch.batch == graph_id
      num_entries = int(node_mask.sum()) * z.shape[-1]
      expected_k = max(1, round(sparsity_ratio * num_entries))
      assert (out[node_mask] != 0).sum().item() == expected_k


# --- variational path -------------------------------------------------------

def test_variational_kl_and_elbo() -> None:
  generator = torch.Generator().manual_seed(3)
  num_channels = 2
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=True,
  )

  x_hat, _, mu, logstd = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                               sparsity_ratio=0.5, temperature=0.5)
  kl = kl_loss(mu, logstd, batch.batch)
  loss, recon, kl2 = elbo_loss(x_hat, batch.x, mu, logstd, batch.batch)

  assert torch.isfinite(kl) and kl.item() >= 0
  assert torch.isclose(kl, kl2)
  assert torch.isfinite(loss) and torch.isfinite(recon)


def test_variational_eval_uses_mean_train_samples() -> None:
  generator = torch.Generator().manual_seed(8)
  num_channels = 2
  batch = _make_batch(num_channels, generator)

  model = SparseVariationalGraphAutoEncoder(
    num_channels=num_channels,
    latent_dim=4,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    num_layers=2,
    is_variational=True,
  )

  model.eval()
  z_eval_1, _, mu1, _ = model.encode(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                     sparsity_ratio=0.5, temperature=0.5)
  z_eval_2, _, mu2, _ = model.encode(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                     sparsity_ratio=0.5, temperature=0.5)
  assert torch.equal(z_eval_1, z_eval_2)
  assert torch.equal(mu1, mu2)

  model.train()
  z_train_1, _, _, _ = model.encode(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                    sparsity_ratio=0.5, temperature=0.5)
  z_train_2, _, _, _ = model.encode(x=batch.x, edge_index=batch.edge_index, batch=batch.batch,
                                    sparsity_ratio=0.5, temperature=0.5)
  assert not torch.equal(z_train_1, z_train_2)
