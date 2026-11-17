#!/usr/bin/env python3
"""Markdown table of success rate and latency for every outputs/<tag>/ run."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
rows = []
ORDER = {"none": 0, "divprune": 1, "random": 2, "pool": 3}
runs = [d for d in root.glob("*/eval_info.json") if not d.parent.name.startswith("_")]


def sort_key(d):
    cfg = json.loads((d.parent / "prune_stats.json").read_text()).get("config", {}) if (d.parent / "prune_stats.json").exists() else {}
    return (ORDER.get(cfg.get("method"), 9), cfg.get("compact", False), -cfg.get("tokens_per_camera", 0))


for d in sorted(runs, key=sort_key):
    info = json.loads(d.read_text())["overall"]
    stats = json.loads((d.parent / "prune_stats.json").read_text()) if (d.parent / "prune_stats.json").exists() else {}
    cfg = stats.get("config", {})
    lo, hi = info["pc_success_ci95"]
    get = lambda k: stats.get(k, {}).get("mean")
    rows.append((d.parent.name, cfg.get("method", "?"), cfg.get("tokens_per_camera", ""), cfg.get("compact", ""),
                 info["pc_success"], lo, hi, info["n_success"], info["n_episodes"], get("prefix_slots"),
                 get("prefill_ms"), get("denoise_ms"), get("total_ms")))
f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
print("| run | method | K | compact | SR % | 95% CI | succ/eps | prefix slots | prefill ms | denoise ms | total ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.1f} | {r[5]:.0f}-{r[6]:.0f} | {r[7]}/{r[8]} | "
          f"{f(r[9], 0)} | {f(r[10])} | {f(r[11])} | {f(r[12])} |")
