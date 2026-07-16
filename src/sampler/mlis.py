import torch


def _allocate_with_capacity(
        weights: list[float],
        total_to_keep: int,
        capacity: int,
) -> list[int]:
    num_scales = len(weights)
    allocation = [0] * num_scales
    active = list(range(num_scales))
    remaining_budget = total_to_keep
    shares = {}

    while active:
        active_weight_sum = sum(weights[i] for i in active)
        if active_weight_sum == 0:
            shares = {i: remaining_budget / len(active) for i in active}
        else:
            shares = {i: remaining_budget * weights[i] / active_weight_sum for i in active}

        saturated = [i for i in active if shares[i] > capacity]
        if not saturated:
            break

        for i in saturated:
            allocation[i] = capacity
            remaining_budget -= capacity
            active.remove(i)

    floors = {i: int(shares[i]) for i in active}
    leftover = remaining_budget - sum(floors.values())
    by_fraction = sorted(active, key=lambda i: shares[i] - floors[i], reverse=True)
    bonus_scales = set(by_fraction[:leftover])

    for i in active:
        allocation[i] = floors[i] + (1 if i in bonus_scales else 0)

    return allocation


def _mlis_single_graph_single_column(
        coeffs: torch.Tensor,
        keep_ratio: float,
) -> torch.Tensor:
    num_nodes, num_filters = coeffs.shape
    total_coeffs = num_nodes * num_filters
    total_to_keep = max(1, int(total_coeffs * keep_ratio))

    squared_coeffs = torch.abs(coeffs) ** 2  # (num_nodes, num_filters)
    scale_energies = squared_coeffs.sum(dim=0)  # (num_filters,)

    weights = scale_energies.tolist()
    if sum(weights) == 0:
        weights = [1.0] * num_filters

    samples_per_scale = _allocate_with_capacity(weights, total_to_keep, capacity=num_nodes)

    mask = torch.zeros_like(coeffs)
    for s in range(num_filters):
        num_samples = samples_per_scale[s]
        if num_samples == 0:
            continue

        p = squared_coeffs[:, s]
        if p.sum() == 0:
            samples = torch.randperm(num_nodes, device=coeffs.device)[:num_samples]
        else:
            samples = torch.multinomial(p, num_samples=num_samples, replacement=False)

        mask[samples, s] = 1.0

    return mask


def multilevel_importance_sampling(
        coeffs: torch.Tensor,
        keep_ratio: float,
        batch: torch.Tensor | None = None,
) -> torch.Tensor:
    num_nodes = coeffs.shape[0]
    num_filters = coeffs.shape[-1]
    column_shape = coeffs.shape[1:-1]  # *S

    num_columns = 1
    for dim_size in column_shape:
        num_columns *= dim_size
    coeffs_flat = coeffs.reshape(num_nodes, num_columns, num_filters)

    if batch is None:
        node_groups = [torch.arange(num_nodes, device=coeffs.device)]
    else:
        batch = batch.to(coeffs.device)
        node_groups = [torch.nonzero(batch == graph_id, as_tuple=True)[0] for graph_id in torch.unique(batch)]

    mask_flat = torch.zeros_like(coeffs_flat)
    for node_idx in node_groups:
        for col in range(num_columns):
            mask_flat[node_idx, col, :] = _mlis_single_graph_single_column(
                coeffs=coeffs_flat[node_idx, col, :],
                keep_ratio=keep_ratio,
            )

    return mask_flat.reshape(coeffs.shape)
