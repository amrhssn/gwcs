import torch


def _allocate_with_capacity_batched(
        weights: torch.Tensor,
        total_to_keep: torch.Tensor,
        capacity: torch.Tensor,
) -> torch.Tensor:
    num_rows, num_filters = weights.shape
    device = weights.device

    weights = weights.to(torch.float64)
    remaining_budget = total_to_keep.to(torch.float64)
    capacity_f = capacity.to(torch.float64)

    active = torch.ones(num_rows, num_filters, dtype=torch.bool, device=device)
    allocation = torch.zeros(num_rows, num_filters, dtype=torch.long, device=device)
    shares = torch.zeros(num_rows, num_filters, dtype=torch.float64, device=device)

    for _ in range(num_filters):
        active_weight_sum = (weights * active).sum(dim=-1)  # (R,)
        num_active = active.sum(dim=-1).to(torch.float64)  # (R,)

        zero_mask = active_weight_sum == 0
        safe_sum = torch.where(zero_mask, torch.ones_like(active_weight_sum), active_weight_sum)
        proportional = remaining_budget.unsqueeze(-1) * weights / safe_sum.unsqueeze(-1)
        uniform = remaining_budget.unsqueeze(-1) / num_active.clamp(min=1).unsqueeze(-1)
        shares = torch.where(zero_mask.unsqueeze(-1), uniform, proportional)
        shares = torch.where(active, shares, torch.zeros_like(shares))

        saturated = active & (shares > capacity_f.unsqueeze(-1))
        allocation = torch.where(saturated, capacity.unsqueeze(-1).expand_as(allocation), allocation)
        remaining_budget = remaining_budget - saturated.sum(dim=-1).to(torch.float64) * capacity_f
        active = active & ~saturated

    floors = torch.floor(shares).long()
    floors = torch.where(active, floors, torch.zeros_like(floors))
    frac = torch.where(active, shares - floors.to(torch.float64), torch.full_like(shares, -1.0))

    leftover = (remaining_budget.round().long() - floors.sum(dim=-1)).clamp(min=0)
    order = torch.argsort(frac, dim=-1, descending=True)
    rank = torch.empty_like(order)
    arange_f = torch.arange(num_filters, device=device).unsqueeze(0).expand(num_rows, num_filters)
    rank.scatter_(-1, order, arange_f)
    bonus = (rank < leftover.unsqueeze(-1)) & active

    allocation = torch.where(active, floors + bonus.long(), allocation)
    return allocation


def _dense_batch(
        coeffs_flat: torch.Tensor,
        batch: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes, num_columns, num_filters = coeffs_flat.shape
    device = coeffs_flat.device

    if batch is None:
        node_counts = torch.tensor([num_nodes], device=device, dtype=torch.long)
        dense_coeffs = coeffs_flat.unsqueeze(0)
        valid_mask = torch.ones(1, num_nodes, dtype=torch.bool, device=device)
        batch_ids = torch.zeros(num_nodes, dtype=torch.long, device=device)
        pos_in_group = torch.arange(num_nodes, device=device)
        return dense_coeffs, valid_mask, node_counts, batch_ids, pos_in_group

    batch = batch.to(device=device, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1  # one sync, O(1) per call

    order = torch.argsort(batch, stable=True)
    sorted_batch = batch[order]
    node_counts = torch.bincount(batch, minlength=num_graphs)
    offsets = torch.zeros(num_graphs, dtype=torch.long, device=device)
    offsets[1:] = node_counts.cumsum(0)[:-1]

    pos_in_sorted = torch.arange(num_nodes, device=device) - offsets[sorted_batch]
    pos_in_group = torch.empty(num_nodes, dtype=torch.long, device=device)
    pos_in_group[order] = pos_in_sorted

    max_nodes = int(node_counts.max().item())  # one sync, O(1) per call

    dense_coeffs = torch.zeros(
        num_graphs, max_nodes, num_columns, num_filters, dtype=coeffs_flat.dtype, device=device,
    )
    dense_coeffs[batch, pos_in_group] = coeffs_flat
    valid_mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool, device=device)
    valid_mask[batch, pos_in_group] = True

    return dense_coeffs, valid_mask, node_counts, batch, pos_in_group


def multilevel_importance_sampling(
        coeffs: torch.Tensor,
        keep_ratio: float,
        batch: torch.Tensor | None = None,
) -> torch.Tensor:
    device = coeffs.device
    num_nodes = coeffs.shape[0]
    num_filters = coeffs.shape[-1]
    column_shape = coeffs.shape[1:-1]  # *S

    num_columns = 1
    for dim_size in column_shape:
        num_columns *= dim_size
    coeffs_flat = coeffs.reshape(num_nodes, num_columns, num_filters)

    dense_coeffs, valid_mask, node_counts, batch_ids, pos_in_group = _dense_batch(coeffs_flat, batch)
    num_graphs, max_nodes = dense_coeffs.shape[0], dense_coeffs.shape[1]

    squared = dense_coeffs.abs() ** 2  # (G, M, C, F); padded slots are exactly 0
    energies = squared.sum(dim=1)  # (G, C, F)
    weights = energies.reshape(num_graphs * num_columns, num_filters)
    zero_rows = weights.sum(dim=-1) == 0
    weights = torch.where(zero_rows.unsqueeze(-1), torch.ones_like(weights), weights)

    capacity_per_row = node_counts.unsqueeze(1).expand(num_graphs, num_columns).reshape(-1)
    total_coeffs_per_row = capacity_per_row * num_filters
    total_to_keep_per_row = (total_coeffs_per_row.to(torch.float64) * keep_ratio).floor().long()
    total_to_keep_per_row = total_to_keep_per_row.clamp(min=1)

    allocation = _allocate_with_capacity_batched(weights, total_to_keep_per_row, capacity_per_row)
    allocation = allocation.reshape(num_graphs, num_columns, num_filters)

    eps = torch.finfo(dense_coeffs.dtype).tiny
    uniform_draws = torch.rand(num_graphs, max_nodes, num_columns, num_filters, device=device)
    keys = torch.log(uniform_draws) / (squared + eps)
    keys = keys.masked_fill(~valid_mask[:, :, None, None], float("-inf"))

    order = torch.argsort(keys, dim=1, descending=True)
    rank = torch.empty_like(order)
    node_arange = torch.arange(max_nodes, device=device).view(1, max_nodes, 1, 1).expand_as(order)
    rank.scatter_(1, order, node_arange)

    mask_dense = (rank < allocation.unsqueeze(1)).to(coeffs.dtype)  # (G, M, C, F)

    mask_flat = mask_dense[batch_ids, pos_in_group]  # (N, C, F)
    return mask_flat.reshape(coeffs.shape)
