from __future__ import annotations

import torch
import torch.nn.functional as F


def token_kmeans(tokens: torch.Tensor, clusters: int, iterations: int = 100) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError("局部特征必须具有 [批, 标记数, 维度] 形状。")
    batch, token_count, _ = tokens.shape
    count = min(int(clusters), token_count)
    if count <= 0:
        raise ValueError("聚类数必须为正数，且局部特征不能为空。")
    initial_indices = torch.linspace(
        0, token_count - 1, count, device=tokens.device
    ).round().long()
    centers = tokens[:, initial_indices, :]
    for _ in range(max(1, int(iterations))):
        assignments = torch.einsum(
            "bnd,bkd->bnk",
            F.normalize(tokens.detach(), dim=-1),
            F.normalize(centers.detach(), dim=-1),
        ).argmax(dim=-1)
        next_centers: list[torch.Tensor] = []
        for index in range(count):
            mask = (assignments == index).to(tokens.dtype).unsqueeze(-1)
            members = mask.sum(dim=1)
            candidate = (tokens * mask).sum(dim=1) / members.clamp_min(1.0)
            next_centers.append(
                torch.where((members > 0).expand_as(candidate), candidate, centers[:, index, :])
            )
        centers = torch.stack(next_centers, dim=1)
    if count < clusters:
        repeated = centers[:, -1:, :].expand(batch, clusters - count, centers.shape[-1])
        centers = torch.cat((centers, repeated), dim=1)
    return F.normalize(centers, dim=-1)


def sinkhorn_plan(
    cost: torch.Tensor,
    *,
    regularization: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-2,
    detach: bool = True,
) -> torch.Tensor:
    if cost.ndim != 3:
        raise ValueError("传输代价必须具有 [批, 行, 列] 形状。")
    batch, rows, columns = cost.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("传输代价矩阵不能为空。")
    plan_cost = cost.detach() if detach else cost
    kernel = torch.exp(-plan_cost / max(float(regularization), 1e-6)).clamp_min(1e-12)
    row_mass = torch.full(
        (batch, rows, 1), 1.0 / rows, dtype=cost.dtype, device=cost.device
    )
    column_mass = torch.full(
        (batch, columns, 1), 1.0 / columns, dtype=cost.dtype, device=cost.device
    )
    left = torch.ones_like(row_mass)
    right = torch.ones_like(column_mass)
    for _ in range(max(1, int(iterations))):
        previous = left
        left = row_mass / torch.bmm(kernel, right).clamp_min(1e-12)
        right = column_mass / torch.bmm(kernel.transpose(1, 2), left).clamp_min(1e-12)
        # 保留参考实现中的设备到主机同步及停止判据，避免悄悄改变算法。
        if float((left - previous).abs().mean().detach().cpu()) < float(tolerance):
            break
    return left * kernel * right.transpose(1, 2)


def local_ot_similarity_from_centers(
    source_centers: torch.Tensor,
    target_centers: torch.Tensor,
    *,
    sinkhorn_iterations: int = 100,
    sinkhorn_regularization: float = 0.1,
    sinkhorn_tolerance: float = 1e-2,
    detach_transport_plan: bool = True,
) -> torch.Tensor:
    cosine = torch.einsum("bnd,bmd->bnm", source_centers, target_centers)
    plan = sinkhorn_plan(
        1.0 - cosine,
        regularization=sinkhorn_regularization,
        iterations=sinkhorn_iterations,
        tolerance=sinkhorn_tolerance,
        detach=detach_transport_plan,
    )
    return (plan * cosine).sum(dim=(1, 2))


def local_ot_similarity(
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor | None = None,
    *,
    target_centers: torch.Tensor | None = None,
    clusters: int = 10,
    cluster_iterations: int = 100,
    sinkhorn_iterations: int = 100,
    sinkhorn_regularization: float = 0.1,
    sinkhorn_tolerance: float = 1e-2,
    detach_transport_plan: bool = True,
) -> torch.Tensor:
    if (target_tokens is None) == (target_centers is None):
        raise ValueError("目标局部特征与已缓存目标中心必须且只能提供一个。")
    source_centers = token_kmeans(source_tokens, clusters, cluster_iterations)
    if target_centers is None:
        assert target_tokens is not None
        target_centers = token_kmeans(target_tokens, clusters, cluster_iterations)
    return local_ot_similarity_from_centers(
        source_centers,
        target_centers,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_regularization=sinkhorn_regularization,
        sinkhorn_tolerance=sinkhorn_tolerance,
        detach_transport_plan=detach_transport_plan,
    )

