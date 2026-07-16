from typing import Literal

import torch

from data.types import ABLSignal


def approximately_bandlimited(
        coeffs: torch.Tensor,
        num_frequency_modes: int,
        signal_type: Literal['lowpass', 'highpass', 'bandpass', 'R-sparse'],
        eigenvector: torch.Tensor,
        noise_level: float | None = None,
        random_generator: torch.Generator | None = None
) -> ABLSignal:
  num_nodes = eigenvector.shape[0]
  if num_frequency_modes > num_nodes:
    raise ValueError(f"num_modes ({num_frequency_modes}) cannot be greater than number of nodes ({num_nodes})")

  if signal_type == 'lowpass':
    selected_indices = torch.arange(num_frequency_modes, device=eigenvector.device)

  elif signal_type == 'highpass':
    selected_indices = torch.arange(num_nodes - num_frequency_modes, num_nodes, device=eigenvector.device)

  elif signal_type == 'bandpass':
    start_idx = (num_nodes - num_frequency_modes) // 2
    selected_indices = torch.arange(start_idx, start_idx + num_frequency_modes, device=eigenvector.device)

  elif signal_type == 'R-sparse':
    perm = torch.randperm(num_nodes, generator=random_generator).to(coeffs.device)
    selected_indices, _ = torch.sort(perm[:num_frequency_modes])

  else:
    raise ValueError(
      f"Invalid signal_type: {signal_type}. Must be one of 'lowpass', 'highpass', 'bandpass', 'R-sparse'")

  U_R = eigenvector[:, selected_indices]
  x_clean = U_R @ coeffs
  if noise_level is not None:
    noise_std = x_clean.std() * noise_level
    noise_std = float(noise_std)
    noise = torch.randn(num_nodes,
                        generator=random_generator,
                        dtype=eigenvector.dtype).to(coeffs.device)
    noise = noise * noise_std
    x = x_clean + noise
  else:
    noise_std = None
    x = x_clean
  x = x.unsqueeze(-1)
  x_clean = x_clean.unsqueeze(-1)

  signal = ABLSignal(x=x,
                     x_clean=x_clean,
                     signal_type=signal_type,
                     num_nodes=num_nodes,
                     num_frequency_modes=num_frequency_modes,
                     selected_indices=selected_indices,
                     coeffs=coeffs,
                     U_R=U_R,
                     noise_level=noise_level,
                     noise_std=noise_std)
  return signal
