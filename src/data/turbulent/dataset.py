from pathlib import Path

import torch
from torch_geometric.data import (
  Data,
  Dataset
)
from torch_geometric.utils import to_torch_coo_tensor

from configs.abl import GraphWaveletTransformConfig
from data.types import TurbulentFieldMetadata, TurbulentGraph, TurbulentSample, TurbulentSampleSignal
from transform.kernels import default_kernels


class TurbulentGraphSignalData(Data):
  def symmetric_normalized_laplacian(self) -> torch.Tensor:
    return to_torch_coo_tensor(edge_index=self.laplacian_index,
                               edge_attr=self.laplacian_weight,
                               size=self.num_nodes,
                               is_coalesced=True)

  def rescaled_symmetric_normalized_laplacian(self) -> torch.Tensor | None:
    if 'rescaled_laplacian_weight' not in self:
      return None
    return to_torch_coo_tensor(edge_index=self.laplacian_index,
                               edge_attr=self.rescaled_laplacian_weight,
                               size=self.num_nodes,
                               is_coalesced=True)


class TurbulentGraphSignalDataset(Dataset):
  def __init__(
          self,
          sample_files: list[Path],
          graph: TurbulentGraph,
          field_metadata: TurbulentFieldMetadata,
          gwt_config: GraphWaveletTransformConfig,
  ) -> None:
    super().__init__()
    self.sample_files = sample_files
    self.graph = graph
    self.field_metadata = field_metadata
    self.gwt_config = gwt_config
    self.edge_index, self.laplacian_index, self.laplacian_weight, self.rescaled_laplacian_weight = \
      _shared_graph_tensors(graph=graph, gwt_config=gwt_config)
    self.num_channels = len(field_metadata.field_names)
    self.num_filters = gwt_config.num_filters
    self.spectral_kernels = default_kernels(minimum_eigenvalue=gwt_config.scaling_kernel_cutoff_eigenvalue,
                                            scales=gwt_config.scales)
    self.storage_path: Path | None = None

  def len(self) -> int:
    return len(self.sample_files)

  def get(self, idx: int) -> TurbulentGraphSignalData:
    signal = _load_signal(self.sample_files[idx])
    data = TurbulentGraphSignalData(
      x=signal.x,
      gwt_coeffs=signal.gwt_coeffs,
      sample_idx=torch.tensor([idx], dtype=torch.long),
      tcool=torch.tensor([signal.tcool], dtype=torch.float32),
      trajectory_id=torch.tensor([signal.trajectory_id], dtype=torch.long),
      timestep=torch.tensor([signal.timestep], dtype=torch.long),
    )
    data.edge_index = self.edge_index
    data.laplacian_index = self.laplacian_index
    data.laplacian_weight = self.laplacian_weight
    if self.rescaled_laplacian_weight is not None:
      data.rescaled_laplacian_weight = self.rescaled_laplacian_weight
    return data

  def get_sample(
          self,
          sample_idx: int
  ) -> TurbulentSample:
    signal = _load_signal(self.sample_files[sample_idx])
    return TurbulentSample(graph=self.graph, x=signal.x, gwt_coeffs=signal.gwt_coeffs, tcool=signal.tcool,
                           trajectory_id=signal.trajectory_id, timestep=signal.timestep)

  def get_samples(
          self,
          sample_idx: torch.Tensor
  ) -> list[TurbulentSample]:
    return [self.get_sample(int(idx)) for idx in sample_idx]


def _load_signal(sample_file: Path) -> TurbulentSampleSignal:
  return torch.load(sample_file, weights_only=False)


def _shared_graph_tensors(
        graph: TurbulentGraph,
        gwt_config: GraphWaveletTransformConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
  laplacian = graph.symmetric_normalized_laplacian.coalesce()
  laplacian_index = laplacian.indices()
  laplacian_weight = laplacian.values()

  rescaled_laplacian_weight = None
  if gwt_config.gwt_mode == "chebyshev":
    a = (gwt_config.spectrum_upper_bound - gwt_config.spectrum_lower_bound) / 2.0
    b = (gwt_config.spectrum_upper_bound + gwt_config.spectrum_lower_bound) / 2.0
    is_diagonal = laplacian_index[0] == laplacian_index[1]
    rescaled_laplacian_weight = laplacian_weight / a - is_diagonal.to(laplacian_weight.dtype) * (b / a)

  return graph.edge_index, laplacian_index, laplacian_weight, rescaled_laplacian_weight
