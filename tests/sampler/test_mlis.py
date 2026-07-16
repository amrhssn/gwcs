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
  num_nodes, keep_ratio = 6, 0.5
  signal_0 = energies_to_coeffs(num_nodes, scale_energies=[100.0, 0.0]).unsqueeze(1)  # (N, 1, F)
  signal_1 = energies_to_coeffs(num_nodes, scale_energies=[0.0, 100.0]).unsqueeze(1)  # (N, 1, F)
  coeffs = torch.stack([signal_0, signal_1], dim=1)  # (N, B=2, C=1, F=2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio)

  assert mask.shape == coeffs.shape
  assert mask[:, 0, 0, 0].sum().item() == num_nodes and mask[:, 0, 0, 1].sum().item() == 0
  assert mask[:, 1, 0, 1].sum().item() == num_nodes and mask[:, 1, 0, 0].sum().item() == 0


def test_batch_argument_separates_graphs_with_different_energy_scales(multilevel_importance_sampling):
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


def test_pytorch_style_batch_axis_of_size_one_stays_explicit(multilevel_importance_sampling):
  num_nodes, keep_ratio = 6, 0.5
  signal = energies_to_coeffs(num_nodes, scale_energies=[100.0, 0.0]).unsqueeze(1)  # (N, C=1, F=2)
  coeffs = signal.unsqueeze(1)  # (N, B=1, C=1, F=2)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio)

  assert mask.shape == coeffs.shape
  assert mask[:, 0, 0, 0].sum().item() == num_nodes and mask[:, 0, 0, 1].sum().item() == 0


def test_single_filter_keeps_budget_and_capacity(multilevel_importance_sampling):
  num_nodes, keep_ratio = 8, 0.5
  coeffs = energies_to_coeffs(num_nodes, scale_energies=[42.0])  # (N, F=1)

  mask = multilevel_importance_sampling(coeffs, keep_ratio=keep_ratio)

  assert mask.shape == coeffs.shape
  assert mask.sum().item() == int(num_nodes * 1 * keep_ratio)
  assert (mask.sum(dim=0) <= num_nodes).all()
