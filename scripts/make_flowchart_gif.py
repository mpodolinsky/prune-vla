#!/usr/bin/env python3
"""Animated flowchart showing where the pruning sits in pi0.5 (writes assets/pruning_flow.gif).

    python scripts/make_flowchart_gif.py [--out assets/pruning_flow.gif]
"""

import argparse
import math
import random
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S = 976, 300, 2  # final size, supersampling factor
INK, MUTED, BOX, BORDER = (17, 24, 39), (107, 114, 128), (243, 244, 246), (156, 163, 175)
ACCENT, ACCENT_BG, ACTIVE = (234, 88, 12), (255, 243, 232), (37, 99, 235)


def farthest_points(n_side: int, k: int, seed: int = 3) -> set[int]:
    """Spread-out subset of a grid (visual stand-in for a diverse selection)."""
    pts = [(i % n_side, i // n_side) for i in range(n_side * n_side)]
    first = random.Random(seed).randrange(len(pts))
    chosen = [first]
    d = [math.dist(p, pts[first]) for p in pts]
    for _ in range(k - 1):
        nxt = max(range(len(pts)), key=lambda i: d[i])
        chosen.append(nxt)
        d = [min(d[i], math.dist(pts[i], pts[nxt])) for i in range(len(pts))]
    return set(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/pruning_flow.gif"))
    ap.add_argument("--k", type=int, default=64)
    args = ap.parse_args()

    f = lambda size: ImageFont.load_default(size=size * S)
    font, small, bold = f(14), f(12), f(15)
    kept = farthest_points(16, args.k)
    order = sorted(set(range(256)) - kept, key=lambda i: random.Random(i).random())  # drop order

    # (x, width, title, subtitle)
    boxes = [(20, 132, "Camera images", "agentview + wrist"),
             (186, 156, "SigLIP + projector", "256 tokens / camera"),
             (376, 206, "", ""),
             (616, 170, "Gemma-2B prefill", "builds the KV cache"),
             (820, 136, "Action expert", "10 denoise steps")]
    cy, bh = 118, 84
    prune_top, prune_bot = 28, 232

    def px(v):
        return int(v * S)

    def frame(active: int, drop: float, tag: str) -> Image.Image:
        im = Image.new("RGB", (W * S, H * S), (255, 255, 255))
        d = ImageDraw.Draw(im)
        for i, (x, w, title, sub) in enumerate(boxes):
            if i == 2:
                top, bot, fill, line, lw = prune_top, prune_bot, ACCENT_BG, ACCENT, 3
            else:
                top, bot = cy - bh // 2, cy + bh // 2
                fill, line, lw = (BOX, BORDER, 2) if i != active else ((232, 240, 254), ACTIVE, 3)
            d.rounded_rectangle([px(x), px(top), px(x + w), px(bot)], radius=px(10), fill=fill, outline=line, width=lw)
            if i != 2:
                d.text((px(x + w / 2), px(cy - 12)), title, fill=INK, font=bold, anchor="mm")
                d.text((px(x + w / 2), px(cy + 12)), sub, fill=MUTED, font=small, anchor="mm")
            if i < len(boxes) - 1:  # arrow
                x0, x1 = x + w + 4, boxes[i + 1][0] - 4
                d.line([px(x0), px(cy), px(x1 - 6), px(cy)], fill=MUTED, width=px(2))
                d.polygon([(px(x1), px(cy)), (px(x1 - 9), px(cy - 5)), (px(x1 - 9), px(cy + 5))], fill=MUTED)
        # prune box content: token grid
        x, w = boxes[2][0], boxes[2][1]
        d.text((px(x + w / 2), px(prune_top + 18)), "DivPrune", fill=ACCENT, font=bold, anchor="mm")
        cell, gx, gy = 8.5, x + w / 2 - 8 * 8.5, prune_top + 36
        n_drop = int(drop * len(order))
        dropped = set(order[:n_drop])
        for i in range(256):
            cx, cyy = gx + (i % 16) * cell + cell / 2, gy + (i // 16) * cell + cell / 2
            if i in dropped:
                col, r = (229, 231, 235), 1.6
            elif drop >= 1.0 and i in kept:
                col, r = ACCENT, 3.2
            else:
                col, r = (75, 85, 99), 2.6
            d.ellipse([px(cx - r), px(cyy - r), px(cx + r), px(cyy + r)], fill=col)
        d.text((px(x + w / 2), px(prune_bot - 22)), f"256 -> K tokens per camera" if drop < 1 else f"256 -> {args.k} tokens per camera",
               fill=INK, font=small, anchor="mm")
        d.text((px(x + w / 2), px(prune_bot - 8)), "(K = 128, 64, 32, ...)", fill=MUTED, font=small, anchor="mm")
        d.text((px(W / 2), px(H - 28)), tag, fill=INK, font=font, anchor="mm")
        d.text((px(W / 2), px(H - 10)), "nothing upstream or downstream is modified; the prefix and KV cache are just shorter",
               fill=MUTED, font=small, anchor="mm")
        return im.resize((W, H), Image.LANCZOS)

    base = "pi0.5: prefix = [ camera tokens | text ]"
    frames = []
    for a in range(2):
        frames += [frame(a, 0.0, base)] * 6
    frames += [frame(2, 0.0, base)] * 4
    frames += [frame(2, t / 10, base) for t in range(1, 11)]
    frames += [frame(2, 1.0, f"prefix shrinks: 565 -> 181 slots (K={args.k}), prefill 47 -> 17 ms")] * 6
    for a in (3, 4):
        frames += [frame(a, 1.0, f"prefix shrinks: 565 -> 181 slots (K={args.k}), prefill 47 -> 17 ms")] * 6
    frames += [frame(-1, 1.0, f"prefix shrinks: 565 -> 181 slots (K={args.k}), prefill 47 -> 17 ms")] * 10

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", "10", "-i", "-", "-vf", "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
           "-loop", "0", str(args.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.asarray(fr).tobytes())
    p.stdin.close()
    p.wait()
    print(f"{args.out}: {args.out.stat().st_size / 1e3:.0f} KB, {len(frames)} frames")


if __name__ == "__main__":
    main()
