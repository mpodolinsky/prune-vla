import math

import torch

from prune_vla.selection import divprune_indices, divprune_select, pool_tokens, random_select


def _min_pairwise_cosine_distance(x: torch.Tensor) -> float:
    x = torch.nn.functional.normalize(x, dim=-1)
    d = 1 - x @ x.T
    d.fill_diagonal_(float("inf"))
    return float(d.min())


def test_shapes_and_sorted_unique_indices():
    x = torch.randn(3, 64, 16)
    idx = divprune_indices(x, 10)
    assert idx.shape == (3, 10)
    assert (idx[:, 1:] > idx[:, :-1]).all()  # ascending => unique and raster order kept
    assert divprune_select(x, 10).shape == (3, 10, 16)


def test_keeps_everything_when_k_covers_all():
    x = torch.randn(2, 8, 4)
    assert torch.equal(divprune_select(x, 8), x)
    assert torch.equal(divprune_select(x, 99), x)


def test_duplicates_are_dropped_first():
    # 4 distinct directions, each repeated 5 times: any k=4 diverse subset takes one of each.
    base = torch.eye(4)
    x = base.repeat_interleave(5, dim=0)[None] + 1e-4 * torch.randn(1, 20, 4)
    kept = divprune_select(x, 4)[0]
    assert set(kept.argmax(-1).tolist()) == {0, 1, 2, 3}


def test_more_diverse_than_random():
    torch.manual_seed(0)
    x = torch.randn(1, 256, 32) + 3.0 * torch.randn(1, 1, 32)  # shared offset makes tokens redundant
    div = _min_pairwise_cosine_distance(divprune_select(x, 32)[0])
    rnd = [_min_pairwise_cosine_distance(random_select(x, 32)[0]) for _ in range(20)]
    assert div > max(rnd)


def test_batch_rows_are_independent():
    torch.manual_seed(1)
    x = torch.randn(2, 64, 8)
    both = divprune_indices(x, 12)
    for i in range(2):
        assert torch.equal(both[i], divprune_indices(x[i : i + 1], 12)[0])


def test_random_select_is_seeded_and_ordered():
    x = torch.randn(2, 64, 8)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a, b = random_select(x, 16, g1), random_select(x, 16, g2)
    assert torch.equal(a, b) and a.shape == (2, 16, 8)


def test_pool_matches_manual_average():
    x = torch.arange(2 * 16 * 3, dtype=torch.float32).view(2, 16, 3)  # 4x4 grid
    p = pool_tokens(x, 4)  # -> 2x2
    grid = x.view(2, 4, 4, 3)
    expect = torch.stack([grid[:, 0:2, 0:2].mean((1, 2)), grid[:, 0:2, 2:4].mean((1, 2)),
                          grid[:, 2:4, 0:2].mean((1, 2)), grid[:, 2:4, 2:4].mean((1, 2))], dim=1)
    assert torch.allclose(p, expect)


def test_pool_rejects_bad_k():
    x = torch.randn(1, 256, 4)
    for bad in (100, 50):
        try:
            pool_tokens(x, bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert pool_tokens(x, 64).shape == (1, 64, 4) and math.isqrt(64) ** 2 == 64
