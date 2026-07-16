import pytest
import torch

from sampler.mlis import multilevel_importance_sampling as mlis_loop
from sampler.mlis_vectorized import multilevel_importance_sampling as mlis_vectorized


def energies_to_coeffs(num_nodes: int, scale_energies: list[float]) -> torch.Tensor:
  """Build a `(num_nodes, num_filters)` tensor with exact per-scale energy
  (sum of squared coefficients), split evenly across nodes."""
  columns = [torch.full((num_nodes,), (energy / num_nodes) ** 0.5) for energy in scale_energies]
  return torch.stack(columns, dim=1)


@pytest.fixture(params=[mlis_loop, mlis_vectorized], ids=["loop", "vectorized"])
def multilevel_importance_sampling(request):
  return request.param
