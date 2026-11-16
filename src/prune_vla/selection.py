"""Token selection rules. All functions take image tokens ``(B, N, D)`` and return ``(B, K, D)``.

``divprune_select`` follows DivPrune (Alvar et al., CVPR 2025, arXiv:2503.02175): greedily pick the K tokens
with the largest minimum pairwise cosine distance (a max-min diversity problem).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def divprune_indices(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Indices ``(B, k)`` of the max-min-diverse subset, in ascending (raster) order.

    1. Seed with the token whose nearest neighbour is farthest away.
    2. Repeatedly add the token whose distance to its nearest *selected* token is largest.
    The cosine-distance matrix is built once; each step only updates a running min-distance vector.
    """
    b, n, _ = tokens.shape
    if k >= n:
        return torch.arange(n, device=tokens.device).expand(b, n)
    if k < 1:
        raise ValueError("k must be >= 1")
    x = F.normalize(tokens.float(), dim=-1)
    dist = (1.0 - x @ x.transpose(1, 2)).clamp_min_(0.0)  # (B, N, N) cosine distance
    rows = torch.arange(b, device=tokens.device)

    eye = torch.eye(n, dtype=torch.bool, device=tokens.device)
    nearest = dist.masked_fill(eye, float("inf")).amin(dim=-1)  # distance to the closest *other* token
    first = nearest.argmax(dim=-1)

    selected = torch.zeros(b, n, dtype=torch.bool, device=tokens.device)
    selected[rows, first] = True
    min_dist = dist[rows, first]  # (B, N): distance to the selected set
    for _ in range(k - 1):
        nxt = min_dist.masked_fill(selected, float("-inf")).argmax(dim=-1)
        selected[rows, nxt] = True
        min_dist = torch.minimum(min_dist, dist[rows, nxt])
    return selected.nonzero()[:, 1].view(b, k)  # row-major nonzero => ascending within each row


def divprune_select(tokens: torch.Tensor, k: int) -> torch.Tensor:
    idx = divprune_indices(tokens, k)
    return tokens.gather(1, idx[..., None].expand(-1, -1, tokens.shape[-1]))


def random_select(tokens: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniformly random subset of K tokens per sample (raster order kept)."""
    b, n, d = tokens.shape
    if k >= n:
        return tokens
    scores = torch.rand(b, n, generator=generator, device=tokens.device)
    idx = scores.topk(k, dim=-1).indices.sort(dim=-1).values
    return tokens.gather(1, idx[..., None].expand(-1, -1, d))


def pool_tokens(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool the square token grid down to K tokens (K must be (side/s)^2 for an integer s)."""
    b, n, d = tokens.shape
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"{n} tokens do not form a square grid")
    if k >= n:
        return tokens
    new_side = math.isqrt(k)
    if new_side * new_side != k or side % new_side:
        raise ValueError(f"cannot pool a {side}x{side} grid to {k} tokens")
    s = side // new_side
    grid = tokens.view(b, side, side, d).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(grid, kernel_size=s, stride=s)
    return pooled.flatten(2).transpose(1, 2).contiguous()
