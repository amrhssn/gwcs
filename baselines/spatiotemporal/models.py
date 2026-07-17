"""
models.py — Spatio-Temporal Baseline Models
=============================================

Lightweight models that operate on full PDE field sequences.
These serve as comparisons against NIGWT / GWCS which operates from
only 1% of the nodes (compressed sensing regime).

Model catalogue
───────────────
1.  ConvLSTM        — 2-D convolutional LSTM on the H×W grid.
                      CNN captures local spatial patterns; LSTM captures
                      temporal dynamics.  Requires the full spatial field.

2.  GRUBaseline     — Flattens nodes to a feature vector, runs a GRU over
                      the temporal sequence.  Simple but captures global
                      temporal dependence.  Memory scales as O(N²), so only
                      feasible for small grids (kolmogorov 128×128).

3.  GraphGRU        — Node-wise GRU where each node's hidden state is
                      updated with both its temporal history and its
                      graph-neighbourhood message.  Works at graph scale.

All models implement a common interface:
    model(x_seq) → x_hat_seq
    where x_seq is (B, T, N) or (B, T, H, W) depending on the model.

NOTE: These baselines use the FULL observed field (no compression).
      The point of comparison with NIGWT is to show that NIGWT achieves
      competitive accuracy from only 1% of the data.
"""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. ConvLSTM  (operates on H×W grid)
# =============================================================================

class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        # Concatenate [input, hidden] along channel dim and project to 4 gate maps
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(
            self,
            x: torch.Tensor,             # (B, C_in, H, W)
            h: torch.Tensor,             # (B, C_h,  H, W)
            c: torch.Tensor,             # (B, C_h,  H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size: int, grid_shape: Tuple[int, int], device):
        H, W = grid_shape
        h = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for PDE field prediction.

    Given a sequence of T input frames, predicts the next T frames
    (or reconstructs a target sequence).

    Input  : (B, T, H, W)  — B batches, T timesteps, H×W spatial
    Output : (B, T, H, W)
    """

    def __init__(
            self,
            in_channels:     int = 1,
            hidden_channels: int = 64,
            num_layers:      int = 2,
            kernel_size:     int = 3,
            out_channels:    int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        cells = []
        for l in range(num_layers):
            c_in = in_channels if l == 0 else hidden_channels
            cells.append(ConvLSTMCell(c_in, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

        self.output_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(
            self,
            x_seq: torch.Tensor,   # (B, T, H, W)
    ) -> torch.Tensor:             # (B, T, H, W)
        B, T, H, W = x_seq.shape
        device = x_seq.device

        # Initialise hidden states
        states = [cell.init_hidden(B, (H, W), device) for cell in self.cells]

        outputs = []
        for t in range(T):
            layer_in = x_seq[:, t, :, :].unsqueeze(1)  # (B, 1, H, W)
            for l, cell in enumerate(self.cells):
                h, c = states[l]
                h, c = cell(layer_in, h, c)
                states[l] = (h, c)
                layer_in = h
            out = self.output_proj(h)        # (B, 1, H, W)
            outputs.append(out)

        return torch.cat(outputs, dim=1).squeeze(2)  # (B, T, H, W) if out_channels=1

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 2. GRUBaseline  (works on flattened nodes — small grids only)
# =============================================================================

class GRUBaseline(nn.Module):
    """
    Simple GRU that treats each timestep's flattened field as a single vector.

    Input  : (B, T, N)    N = H×W nodes
    Output : (B, T, N)

    ⚠  Memory: O(B × T × N × hidden_dim) — use only for N ≤ 16384 (128×128).
    """

    def __init__(
            self,
            num_nodes:   int,
            hidden_dim:  int = 256,
            num_layers:  int = 2,
            dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(num_nodes, hidden_dim)
        self.gru         = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, N)
        h = self.input_proj(x_seq)             # (B, T, D)
        h, _ = self.gru(h)                     # (B, T, D)
        return self.output_proj(h)             # (B, T, N)


# =============================================================================
# 3. GraphGRU  (GNN message-passing + GRU at each node)
# =============================================================================

class GraphGRUCell(nn.Module):
    """
    Graph GRU cell where each node aggregates neighbour messages before
    updating its GRU hidden state.

    At each timestep:
        m_i = mean_{j ∈ N(i)} W_msg · [h_j || x_j]
        h_i = GRUCell(x_i + m_i, h_i)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Message projection: (hidden_dim + input_dim) → hidden_dim
        self.msg_proj = nn.Linear(hidden_dim + input_dim, hidden_dim)
        # GRU update
        self.gru_cell = nn.GRUCell(input_dim + hidden_dim, hidden_dim)

    def forward(
            self,
            x: torch.Tensor,            # (N, input_dim) — current frame
            h: torch.Tensor,            # (N, hidden_dim) — previous hidden
            edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:                  # (N, hidden_dim)
        row, col = edge_index[0], edge_index[1]
        N = x.shape[0]

        # Message: for each edge (j→i), compute message from j
        src_feat = torch.cat([h[col], x[col]], dim=-1)  # (E, H+D)
        msgs     = self.msg_proj(src_feat)               # (E, H)

        # Aggregate: mean pooling
        agg = torch.zeros(N, self.hidden_dim, device=x.device)
        agg.scatter_add_(0, row.unsqueeze(1).expand_as(msgs), msgs)
        deg = torch.bincount(row, minlength=N).float().clamp(min=1).unsqueeze(1)
        agg = agg / deg

        # GRU update: input is [x_i, agg_i]
        gru_in = torch.cat([x, agg], dim=-1)   # (N, D+H)
        h_new  = self.gru_cell(gru_in, h)      # (N, H)
        return h_new


class GraphGRU(nn.Module):
    """
    Graph GRU for PDE field prediction on unstructured graphs.

    Each timestep, node features are updated via graph message-passing
    followed by a GRU update.  The decoder then projects back to the
    physical field.

    Input  : x_seq (T, N, C_in), edge_index (2, E)
    Output : x_hat_seq (T, N, C_out)
    """

    def __init__(
            self,
            input_dim:  int = 1,
            hidden_dim: int = 64,
            output_dim: int = 1,
            num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.cells = nn.ModuleList([
            GraphGRUCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
            self,
            x_seq: torch.Tensor,        # (T, N, C_in)
            edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:                  # (T, N, C_out)
        T, N, _ = x_seq.shape
        device = x_seq.device

        # Initialise hidden states
        hidden_states = [
            torch.zeros(N, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(T):
            x_t = self.input_proj(x_seq[t])   # (N, H)
            for l, cell in enumerate(self.cells):
                x_t = cell(x_t, hidden_states[l], edge_index)
                hidden_states[l] = x_t
            out_t = self.output_proj(x_t)      # (N, C_out)
            outputs.append(out_t)

        return torch.stack(outputs, dim=0)   # (T, N, C_out)
