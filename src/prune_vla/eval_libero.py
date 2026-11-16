"""Evaluate pi0.5 (``lerobot/pi05_libero``) on LIBERO-10 with visual-token pruning, via LeRobot's own eval loop.

    uv run python -m prune_vla.eval_libero --method divprune --tokens-per-camera 64 --compact \
        --n-episodes 5 --profile

Arguments LeRobot's ``lerobot-eval`` understands can be appended after ``--`` (e.g. ``-- --eval.n_episodes=20``).
Writes ``eval_info.json`` (LeRobot), ``prune_stats.json`` (timings, token counts) and ``prune_run.json`` to
``--output-dir`` (default ``outputs/<method>_k<K>[_compact]``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def prepare_libero_env() -> None:
    """Headless rendering and a repo-local LIBERO config (so the first import never prompts)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    cfg_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero")))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / "config.yaml"
    if not cfg.exists():
        root = REPO / "submodules" / "LIBERO" / "libero" / "libero"
        cfg.write_text(
            "\n".join(
                f"{k}: {v}"
                for k, v in {
                    "benchmark_root": root,
                    "bddl_files": root / "bddl_files",
                    "init_states": root / "init_files",
                    "datasets": root.parent / "datasets",
                    "assets": root / "assets",
                }.items()
            )
            + "\n"
        )


ASSETS = Path(__file__).resolve().parent / "assets"


def load_norm_stats(path: Path):
    """openpi-format ``norm_stats.json`` -> LeRobot ``dataset_stats`` (observation.state / action)."""
    import torch

    raw = json.loads(Path(path).read_text())["norm_stats"]
    key = {"state": "observation.state", "actions": "action"}
    return {key[k]: {s: torch.tensor(v, dtype=torch.float32) for s, v in stats.items()} for k, stats in raw.items()}


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["none", "divprune", "random", "pool"], default="none")
    ap.add_argument("--tokens-per-camera", type=int, default=256, help="K kept per real camera (256 = all)")
    ap.add_argument("--compact", action="store_true", help="drop prefix slots that are pure padding")
    ap.add_argument("--profile", action="store_true", help="time prefill/denoise per policy call")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--n-episodes", type=int, default=5, help="episodes per task")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None, help="LIBERO-10 task ids (default: all 10)")
    ap.add_argument("--policy-path", default="lerobot/pi05_libero")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--norm-stats", type=Path, default=ASSETS / "pi05_libero_norm_stats.json",
                    help="quantile stats of the pi05_libero checkpoint (the LeRobot copy ships none)")
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, rest = parse_args(sys.argv[1:] if argv is None else argv)
    rest = [r for r in rest if r != "--"]
    prepare_libero_env()

    from prune_vla.patch import PruneConfig, apply_pruning

    tag = f"{args.method}_k{args.tokens_per_camera}" + ("_compact" if args.compact else "")
    out = args.output_dir or REPO / "outputs" / tag
    cfg = PruneConfig(method=args.method, tokens_per_camera=args.tokens_per_camera, compact=args.compact,
                      seed=args.seed, profile=args.profile)

    import lerobot.scripts.lerobot_eval as le

    handle = {}
    original_make_policy = le.make_policy

    def make_policy(*a, **kw):
        policy = original_make_policy(*a, **kw)
        handle["h"] = apply_pruning(policy, cfg)
        return policy

    le.make_policy = make_policy

    # `lerobot/pi05_libero` ships processor files without normalisation statistics, which silently turns state and
    # action (un)normalisation into a no-op. Rebuild the processors from the policy config with openpi's stats.
    stats = load_norm_stats(args.norm_stats)
    original_make_processors = le.make_pre_post_processors

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        return original_make_processors(policy_cfg, dataset_stats=stats)

    le.make_pre_post_processors = make_pre_post_processors

    lerobot_args = [
        f"--policy.path={args.policy_path}",
        f"--policy.device={args.device}",
        "--env.type=libero",
        "--env.task=libero_10",
        "--eval.batch_size=1",
        f"--eval.n_episodes={args.n_episodes}",
        f"--seed={args.seed}",
        f"--output_dir={out}",
        *( [f"--env.task_ids=[{','.join(map(str, args.task_ids))}]"] if args.task_ids else [] ),
        *rest,
    ]
    sys.argv = ["lerobot-eval", *lerobot_args]
    Path(out).mkdir(parents=True, exist_ok=True)
    try:
        le.main()
    finally:
        if "h" in handle:
            handle["h"].save(Path(out) / "prune_stats.json")
        (Path(out) / "prune_run.json").write_text(json.dumps({"prune": asdict(cfg), "lerobot_args": lerobot_args}, indent=2))


if __name__ == "__main__":
    main()
