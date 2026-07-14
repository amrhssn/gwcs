import torch


def flatten_trailing_dims(
        signal: torch.Tensor
) -> tuple[torch.Tensor, torch.Size]:
  trailing_shape = signal.shape[1:]
  num_nodes = signal.shape[0]
  num_columns = 1
  for dim in trailing_shape:
    num_columns *= dim
  return signal.reshape(num_nodes, num_columns), trailing_shape


def unflatten_trailing_dims(
        flat: torch.Tensor,
        trailing_shape: torch.Size
) -> torch.Tensor:
  num_nodes = flat.shape[0]
  extra_shape = flat.shape[2:]
  return flat.reshape(num_nodes, *trailing_shape, *extra_shape)
