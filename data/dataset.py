import torch
from torch_geometric.data import (
  Data,
  Dataset,
  InMemoryDataset
)

from data.types import (
  ABLGraphSignalSample,
  GenerationConfig
)


class ABLGraphSignalDataset(InMemoryDataset):
  def __init__(
          self,
          samples: list[ABLGraphSignalSample],
          configs: GenerationConfig
  ) -> None:
    super().__init__()
    self.samples = samples
    self.configs = configs
    data_list = [sample_to_data(sample, sample_idx) for sample_idx, sample in enumerate(self.samples)]
    self.data, self.slices = self.collate(data_list)

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
        sample_idx: int
) -> Data:
  data = Data(x=sample.signal.x,
              edge_index=sample.graph.edge_index,
              gwt_coeffs=sample.gwt_coeffs,
              sample_idx=torch.tensor([sample_idx], dtype=torch.long, device=sample.signal.x.device))
  return data


def split_dataset(
        dataset: ABLGraphSignalDataset,
        train_ratio: float,
        val_ratio: float,
        random_generator: torch.Generator | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
  num_train = int(train_ratio * len(dataset))
  num_val = int(val_ratio * len(dataset))
  perm = torch.randperm(len(dataset), generator=random_generator)
  train_dataset = dataset.index_select(perm[:num_train])
  val_dataset = dataset.index_select(perm[num_train:num_train + num_val])
  test_dataset = dataset.index_select(perm[num_train + num_val:])
  return train_dataset, val_dataset, test_dataset
