from pathlib import Path

import torch
from torch_geometric.data import (
  Data,
  InMemoryDataset
)
from torch_geometric.utils import to_torch_coo_tensor

from data.types import (
  ABLGraphSignalSample,
  GenerationConfig,
  GraphWaveletTransformConfig
)
from transform.kernels import default_kernels


class ABLGraphSignalData(Data):
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


class ABLGraphSignalDataset(InMemoryDataset):
  def __init__(
          self,
          samples: list[ABLGraphSignalSample],
          configs: GenerationConfig
  ) -> None:
    super().__init__()
    self.samples = samples
    self.configs = configs
    data_list = [
      sample_to_data(sample=sample, sample_idx=sample_idx, gwt_config=configs.gwt_config)
      for sample_idx, sample in enumerate(self.samples)
    ]
    self.data, self.slices = self.collate(data_list)
    self.num_channels = self._data.x.shape[-1]
    self.num_filters = self._data.gwt_coeffs.shape[-1]
    self.spectral_kernels = default_kernels(minimum_eigenvalue=configs.gwt_config.scaling_kernel_cutoff_eigenvalue,
                                            scales=configs.gwt_config.scales)
    self.storage_path: Path | None = None

  def get_sample(
          self,
          sample_idx: int
  ) -> ABLGraphSignalSample:
    return self.samples[sample_idx]

  def get_samples(
          self,
          sample_idx: torch.Tensor
  ) -> list[ABLGraphSignalSample]:
    return [self.get_sample(int(idx)) for idx in sample_idx]


def sample_to_data(
        sample: ABLGraphSignalSample,
        sample_idx: int,
        gwt_config: GraphWaveletTransformConfig
) -> ABLGraphSignalData:
  laplacian = sample.graph.symmetric_normalized_laplacian.coalesce()
  laplacian_index = laplacian.indices()
  laplacian_weight = laplacian.values()

  data = ABLGraphSignalData(x=sample.signal.x,
                            edge_index=sample.graph.edge_index,
                            gwt_coeffs=sample.gwt_coeffs,
                            sample_idx=torch.tensor([sample_idx], dtype=torch.long, device=sample.signal.x.device),
                            laplacian_index=laplacian_index,
                            laplacian_weight=laplacian_weight)

  if gwt_config.gwt_mode == "chebyshev":
    a = (gwt_config.spectrum_upper_bound - gwt_config.spectrum_lower_bound) / 2.0
    b = (gwt_config.spectrum_upper_bound + gwt_config.spectrum_lower_bound) / 2.0
    is_diagonal = laplacian_index[0] == laplacian_index[1]
    data.rescaled_laplacian_weight = laplacian_weight / a - is_diagonal.to(laplacian_weight.dtype) * (b / a)

  return data


def split_dataset(
        dataset: ABLGraphSignalDataset,
        train_ratio: float,
        val_ratio: float,
        random_generator: torch.Generator | None = None,
) -> tuple[ABLGraphSignalDataset, ABLGraphSignalDataset, ABLGraphSignalDataset]:
  num_train = int(train_ratio * len(dataset))
  num_val = int(val_ratio * len(dataset))
  perm = torch.randperm(len(dataset), generator=random_generator)
  train_dataset = dataset.index_select(perm[:num_train])
  val_dataset = dataset.index_select(perm[num_train:num_train + num_val])
  test_dataset = dataset.index_select(perm[num_train + num_val:])
  return train_dataset, val_dataset, test_dataset
