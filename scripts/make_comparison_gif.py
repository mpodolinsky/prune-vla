#!/usr/bin/env python3
"""Side-by-side GIF of the same LIBERO-10 episode under different pruning settings.

    python scripts/make_comparison_gif.py --task 1 --episode 0 \
        --run none_k256_compact "no pruning" --run divprune_k128_compact "DivPrune K=128" \
        --run divprune_k64_compact "DivPrune K=64" --out assets/task1_ep0.gif

Uses the rollout videos in outputs/<run>/videos/libero_10_<task>/ and the success flags in eval_info.json.
Every panel is cut at the end of the longest successful run (+ a short hold); finished runs freeze on their last frame.
"""

import argparse
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--run", nargs=2, action="append", metavar=("DIR", "LABEL"), required=True)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", type=int, default=240, help="panel side in pixels")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th frame (video is 20 fps)")
    ap.add_argument("--hold-s", type=float, default=1.5)
    args = ap.parse_args()

    runs = []
    for name, label in args.run:
        info = json.loads((args.outputs / name / "eval_info.json").read_text())
        ok = next(t for t in info["per_task"] if t["task_id"] == args.task)["metrics"]["successes"][args.episode]
        frames = [f for f in imageio.get_reader(args.outputs / name / "videos" / f"libero_10_{args.task}" / f"eval_episode_{args.episode}.mp4")]
        runs.append((label, bool(ok), frames[:: args.stride]))
    fps = 20 // args.stride
    n = max(len(f) for _, ok, f in runs if ok) + int(args.hold_s * fps) if any(ok for _, ok, _ in runs) else max(len(f) for *_, f in runs)

    font = ImageFont.load_default(size=14)
    side, head = args.panel, 24
    canvas_w = side * len(runs)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{canvas_w}x{side + head}", "-r", str(fps), "-i", "-",
           "-vf", "split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
           "-loop", "0", str(args.out)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        canvas = Image.new("RGB", (canvas_w, side + head), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        for j, (label, ok, frames) in enumerate(runs):
            frame = Image.fromarray(frames[min(i, len(frames) - 1)]).resize((side, side), Image.LANCZOS)
            canvas.paste(frame, (j * side, head))
            done = i >= len(frames) - 1 and ok
            tag = "success" if done else ("failed (520 steps)" if not ok else "")
            draw.text((j * side + 6, 4), f"{label}  {tag}".strip(), fill=(0, 220, 120) if done else (235, 235, 235), font=font)
        proc.stdin.write(np.asarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB, {n} frames at {fps} fps")


if __name__ == "__main__":
    main()
