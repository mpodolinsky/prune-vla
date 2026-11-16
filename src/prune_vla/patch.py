"""Instance-level patches that shorten the visual part of pi0.5's prefix (LeRobot ``PI05Policy``).

Where the pruning happens::

    images -> SigLIP -> projector -> [ prune each real camera: 256 -> K tokens ] -> embed_prefix -> Gemma + expert

Nothing in the model is modified; two methods of ``policy.model`` are wrapped on the instance and can be restored.
The kept tokens get contiguous positions (``cumsum(pad_mask) - 1``, LeRobot's own scheme), and the action expert's
KV cache is simply shorter.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .selection import divprune_select, pool_tokens, random_select

Method = Literal["none", "divprune", "random", "pool"]


@dataclass
class PruneConfig:
    method: Method = "none"
    tokens_per_camera: int = 256  # K kept per real camera (SigLIP produces 256 at 224x224)
    compact: bool = False  # physically drop prefix slots that are padding for the whole batch
    seed: int = 0  # for method="random"
    profile: bool = False  # time prefill / denoise / whole call (adds cuda syncs)


class PruneStats:
    """Per-call measurements collected when ``PruneConfig.profile`` is set."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []

    def add(self, **kw: float) -> None:
        self.rows.append(kw)

    def summary(self, warmup: int = 3) -> dict:
        rows = self.rows[warmup:] if len(self.rows) > warmup else self.rows
        out: dict = {"n_calls": len(self.rows), "n_warmup_skipped": len(self.rows) - len(rows)}
        for key in sorted({k for r in rows for k in r}):
            vals = [r[key] for r in rows if key in r]
            out[key] = {"mean": statistics.fmean(vals), "median": statistics.median(vals)}
        return out


class PruneHandle:
    def __init__(self, policy, cfg: PruneConfig):
        self.policy, self.cfg, self.stats = policy, cfg, PruneStats()
        self._originals: list[tuple[object, str, object]] = []
        self._call: dict[str, float] = {}

    def _patch(self, obj, name: str, fn) -> None:
        self._originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, fn)

    def restore(self) -> None:
        for obj, name, orig in reversed(self._originals):
            setattr(obj, name, orig)
        self._originals.clear()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"config": asdict(self.cfg), **self.stats.summary()}, indent=2))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def apply_pruning(policy, cfg: PruneConfig) -> PruneHandle:
    """Patch ``policy.model`` (a ``PI05Pytorch``) in place and return a handle (``restore()`` undoes it)."""
    model = policy.model
    h = PruneHandle(policy, cfg)
    gen = None
    if cfg.method == "random":
        device = next(model.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.seed)

    if cfg.method != "none":
        orig_embed_images = model._embed_images
        k = cfg.tokens_per_camera

        def select(emb: torch.Tensor) -> torch.Tensor:
            if k >= emb.shape[1]:
                return emb
            if cfg.method == "divprune":
                return divprune_select(emb, k)
            if cfg.method == "random":
                return random_select(emb, k, gen)
            if cfg.method == "pool":
                return pool_tokens(emb, k)
            raise ValueError(f"unknown method {cfg.method!r}")

        def embed_images(images, img_masks):
            outs = orig_embed_images(images, img_masks)
            # A camera slot that is absent for the whole batch (pi0.5's "empty camera") is masked out anyway.
            return [select(e) if bool(m.any()) else e for e, m in zip(outs, img_masks, strict=True)]

        h._patch(model, "_embed_images", embed_images)

    if cfg.compact or cfg.profile:
        orig_embed_prefix = model.embed_prefix

        def embed_prefix(*args, **kwargs):
            if cfg.profile:
                _sync()
                t0 = time.perf_counter()
            embs, pad, att = orig_embed_prefix(*args, **kwargs)
            if cfg.compact:
                keep = pad.any(dim=0)  # slot is padding for every sample -> contributes nothing
                if not bool(keep.all()):
                    embs, pad, att = embs[:, keep], pad[:, keep], att[:, keep]
            if cfg.profile:
                _sync()
                h._call["embed_prefix_ms"] = 1000 * (time.perf_counter() - t0)
                h._call["prefix_slots"] = float(embs.shape[1])
                h._call["prefix_valid_tokens"] = float(pad.sum(dim=1).float().mean())
            return embs, pad, att

        h._patch(model, "embed_prefix", embed_prefix)

    if cfg.profile:
        orig_forward = model.paligemma_with_expert.forward
        orig_sample = model.sample_actions

        def forward(*args, **kwargs):
            embeds = kwargs.get("inputs_embeds")
            is_prefill = embeds is not None and embeds[1] is None
            _sync()
            t0 = time.perf_counter()
            out = orig_forward(*args, **kwargs)
            _sync()
            ms = 1000 * (time.perf_counter() - t0)
            if is_prefill:
                h._call["prefill_ms"] = ms
            else:
                h._call["denoise_ms"] = h._call.get("denoise_ms", 0.0) + ms
            return out

        def sample_actions(*args, **kwargs):
            h._call = {}
            _sync()
            t0 = time.perf_counter()
            out = orig_sample(*args, **kwargs)
            _sync()
            h._call["total_ms"] = 1000 * (time.perf_counter() - t0)
            h.stats.add(**h._call)
            return out

        h._patch(model.paligemma_with_expert, "forward", forward)
        h._patch(model, "sample_actions", sample_actions)
    return h
