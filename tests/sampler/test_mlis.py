"""
Correctness checks for `multilevel_importance_sampling`.

Parametrized (via the `multilevel_importance_sampling` fixture in
`conftest.py`) over both the loop-based implementation (`sampler.mlis`) and
the batched, sync-free rewrite (`sampler.mlis_vectorized`), so every test
below runs once per implementation and confirms they make identical
guarantees:
1. Basic shape/dtype/budget sanity on the simplest case (single graph, no
   extra axes, no `batch`).
2. The capacity-allocation bug this module was written to fix: a skewed
   per-scale energy distribution must not lose budget to the capacity clamp.
3. The all-zero-energy fallback doesn't crash and still hits the budget.
4. Extra axes between the node axis and the filter axis (channels, or a
   batch of signals sharing one graph) are sampled completely
   independently of each other -- no energy pooling, no shared budget.
5. `batch` correctly separates multiple graphs concatenated along the node
   axis: each graph gets its own budget from its own energy, regardless of
   how much energy the other graphs in the batch have.
6. `batch` and extra axes compose: each (graph, column) pair is independent.
7. The `keep_ratio == 1.0` closed form (every scale must hit exactly its
   node-count capacity, so the mask is all ones).
"""
import torch

from tests.sampler.conftest import energies_to_coeffs


def test_basic_shape_and_budget(multilevel_importance_sampling):
  torch.manual_seed(0)
  coeffs = torch.randn(10, 4)
  mask = multilevel_importance_sampling(coeffs, keep_ratio=0.5)

  assert mask.shape == coeffs.shape
  assert set(mask.unique().tolist()) <= {0.0, 1.0}
  assert mask.sum().item() == int(10 * 4 * 0.5)
  assert (mask.sum(dim=0) <= 10).all()  # no scale exceeds its node capacity


def test_capacity_allocation_does_not_drop_budget(multilevel_importance_sampling):
  # One scale's energy share alone would demand more samples than there are
  # nodes -- the old clamp-after-allocation code silently discarded that
  # excess instead of handing it to the other scales.
  num_nodes = 3
  coeffs = energies_to_coeffs(num_nodes, scale_energies=[90.0, 4.0, 3.0, 3.0])
  total_coeffs = num_nodes * 4
  mask = multilevel_importance_sampling(coeffs, keep_ratio=0.5)

  assert mask.sum().item() == int(total_coeffs * 0.5)  # == 6, not fewer
  assert (mask.sum(dim=0) <= num_nodes).all()


def test_all_zero_energy_fallback(multilevel_importance_sampling):
  coeffs = torch.zeros(6, 3)
  mask = multilevel_importance_sampling(coeffs, keep_ratio=0.5)

  assert mask.sum().item() == int(6 * 3 * 0.5)
  assert (mask.sum(dim=0) <= 6).all()


def test_extra_axes_are_sampled_independently(multilevel_importance_sampling):
  # Two channels on one shared graph: channel 0's energy lives entirely on
  # scale 0, channel 1's entirely on scale 1. If channels were pooled
  # together (instead of sampled independently), both channels' masks
  # would look the same; they must not.
  num_nodes, keep_ratio = 6, 0.5
  channel_0 = energies_to_coeffs(num_nodes, scale_energies=[100.0, 0.0])
  channel_1 = energies_to_coeffs(num_nodes, scale_energies=[0.0, 100.0])
  coeffs = torch.stack([channel_0, channel_1], dim=1)  # (N, C=2, F=2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio)

  expected_total = int(num_nodes * 2 * keep_ratio)
  assert mask[:, 0, :].sum().item() == expected_total
  assert mask[:, 1, :].sum().item() == expected_total
  # channel 0 keeps only scale 0; channel 1 keeps only scale 1
  assert mask[:, 0, 0].sum().item() == num_nodes and mask[:, 0, 1].sum().item() == 0
  assert mask[:, 1, 1].sum().item() == num_nodes and mask[:, 1, 0].sum().item() == 0


def test_pytorch_style_batch_axis_is_independent_without_batch_arg(multilevel_importance_sampling):
  # A batch of B=2 signals sharing one graph (PyTorch convention): B lands
  # in *S right next to the channel axis. No `batch` argument is passed --
  # this kind of batching must "just work" as extra independent columns.
  num_nodes, keep_ratio = 6, 0.5
  signal_0 = energies_to_coeffs(num_nodes, scale_energies=[100.0, 0.0]).unsqueeze(1)  # (N, 1, F)
  signal_1 = energies_to_coeffs(num_nodes, scale_energies=[0.0, 100.0]).unsqueeze(1)  # (N, 1, F)
  coeffs = torch.stack([signal_0, signal_1], dim=1)  # (N, B=2, C=1, F=2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio)

  assert mask.shape == coeffs.shape
  assert mask[:, 0, 0, 0].sum().item() == num_nodes and mask[:, 0, 0, 1].sum().item() == 0
  assert mask[:, 1, 0, 1].sum().item() == num_nodes and mask[:, 1, 0, 0].sum().item() == 0


def test_batch_argument_separates_graphs_with_different_energy_scales(multilevel_importance_sampling):
  # Two PyG-concatenated graphs of different sizes and wildly different
  # coefficient magnitudes. Without `batch`, the huge-energy graph would
  # dominate sampling and the tiny-energy graph would be starved.
  n1, n2, keep_ratio = 4, 6, 0.5
  graph_1 = energies_to_coeffs(n1, scale_energies=[1e-6, 1e-6, 1e-6])  # tiny energy
  graph_2 = energies_to_coeffs(n2, scale_energies=[1e6, 1e6, 1e6])  # huge energy
  coeffs = torch.cat([graph_1, graph_2], dim=0)  # (N=10, F=3)
  batch = torch.tensor([0] * n1 + [1] * n2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio, batch=batch)

  assert mask[:n1].sum().item() == int(n1 * 3 * keep_ratio)
  assert mask[n1:].sum().item() == int(n2 * 3 * keep_ratio)
  assert (mask[:n1].sum(dim=0) <= n1).all()
  assert (mask[n1:].sum(dim=0) <= n2).all()


def test_batch_and_channels_compose(multilevel_importance_sampling):
  n1, n2, keep_ratio = 4, 6, 0.5
  torch.manual_seed(1)
  coeffs = torch.randn(n1 + n2, 2, 3)  # (N, C=2, F=3)
  batch = torch.tensor([0] * n1 + [1] * n2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio, batch=batch)

  for graph_slice, num_nodes in [(slice(0, n1), n1), (slice(n1, n1 + n2), n2)]:
    for c in range(2):
      assert mask[graph_slice, c, :].sum().item() == int(num_nodes * 3 * keep_ratio)
      assert (mask[graph_slice, c, :].sum(dim=0) <= num_nodes).all()


def test_keep_ratio_one_keeps_everything(multilevel_importance_sampling):
  torch.manual_seed(2)
  coeffs = torch.randn(7, 5)
  mask = multilevel_importance_sampling(coeffs, keep_ratio=1.0)

  assert torch.equal(mask, torch.ones_like(coeffs))
