"""Needs a CUDA GPU and the lerobot/pi05_libero checkpoint (cached or downloadable). Run with ``pytest -m gpu``."""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def policy_and_batch():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.from_pretrained("lerobot/pi05_libero").to("cuda").eval()
    g = torch.Generator("cuda").manual_seed(0)
    bsz = 1
    batch = {}
    for name in ("image", "image2"):
        batch[f"observation.images.{name}"] = torch.rand(bsz, 3, 224, 224, device="cuda", generator=g)
    batch["observation.state"] = torch.randn(bsz, 8, device="cuda", generator=g)
    tokens = torch.zeros(bsz, 200, dtype=torch.long, device="cuda")
    tokens[:, :40] = torch.randint(5, 1000, (bsz, 40), device="cuda", generator=g)
    mask = torch.zeros(bsz, 200, dtype=torch.bool, device="cuda")
    mask[:, :40] = True
    batch["observation.language.tokens"] = tokens
    batch["observation.language.attention_mask"] = mask
    noise = torch.randn(bsz, policy.config.chunk_size, policy.config.max_action_dim, device="cuda", generator=g)
    return policy, batch, noise


def _actions(policy, batch, noise):
    with torch.no_grad():
        return policy.predict_action_chunk(batch, noise=noise)


def test_unpruned_patch_is_a_no_op(policy_and_batch):
    from prune_vla import PruneConfig, apply_pruning

    policy, batch, noise = policy_and_batch
    ref = _actions(policy, batch, noise)
    h = apply_pruning(policy, PruneConfig(method="divprune", tokens_per_camera=256))
    try:
        assert torch.equal(_actions(policy, batch, noise), ref)
    finally:
        h.restore()


def test_compaction_does_not_change_actions(policy_and_batch):
    from prune_vla import PruneConfig, apply_pruning

    policy, batch, noise = policy_and_batch
    ref = _actions(policy, batch, noise)
    h = apply_pruning(policy, PruneConfig(compact=True, profile=True))
    try:
        out = _actions(policy, batch, noise)
        slots = h.stats.rows[-1]["prefix_slots"]
    finally:
        h.restore()
    assert slots < 968
    assert torch.allclose(out, ref, atol=2e-3, rtol=1e-3), float((out - ref).abs().max())


def test_pruned_prefix_is_shorter_and_actions_stay_finite(policy_and_batch):
    from prune_vla import PruneConfig, apply_pruning

    policy, batch, noise = policy_and_batch
    h = apply_pruning(policy, PruneConfig(method="divprune", tokens_per_camera=64, compact=True, profile=True))
    try:
        out = _actions(policy, batch, noise)
        valid = h.stats.rows[-1]["prefix_valid_tokens"]
    finally:
        h.restore()
    assert torch.isfinite(out).all()
    assert valid == 2 * 64 + 40  # two real cameras at K=64 plus the 40 real language tokens
