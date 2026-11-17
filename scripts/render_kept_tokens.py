#!/usr/bin/env python3
"""Show which image patches survive pruning, from a ``selection.npz`` written by ``--record-selection``.

Each kept token is one 14x14 patch of the 224x224 image the model received (SigLIP's 16x16 grid, raster order).
Dropped patches are darkened. This shows *where* tokens are kept; it is NOT what the model "sees": SigLIP's attention
has already mixed information from every patch into every token before pruning, so kept tokens still carry some
information about the dropped regions.

    # same first frame, different methods
    python scripts/render_kept_tokens.py compare --call 0 --out assets/kept_compare.png \
        outputs/_sel_divprune_k128 "DivPrune K=128" outputs/_sel_divprune_k64 "DivPrune K=64" outputs/_sel_random_k64 "random K=64"
    # one run over time
    python scripts/render_kept_tokens.py gif outputs/_sel_divprune_k64 --label "DivPrune K=64" --out assets/kept_over_time.gif
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATCH, GRID = 14, 16
CAMS = ("agentview", "wrist")


def kept_view(img: np.ndarray, kept: np.ndarray, dim: float = 0.2) -> np.ndarray:
    mask = np.zeros(GRID * GRID, dtype=bool)
    mask[kept] = True
    up = np.kron(mask.reshape(GRID, GRID), np.ones((PATCH, PATCH), dtype=bool))[..., None]
    return np.where(up, img, (img * dim).astype(np.uint8)).astype(np.uint8)


def load(run: Path):
    z = np.load(Path(run) / "selection.npz")
    return z["images"], z["kept"]  # (N, 2, 224, 224, 3), (N, 2, K)


def row(images, kept, call: int) -> np.ndarray:
    parts = []
    for c in range(images.shape[1]):
        parts += [images[call, c], kept_view(images[call, c], kept[call, c])]
    return np.concatenate(parts, axis=1)


def labelled(canvas: Image.Image, text: str, x: int, y: int, size: int = 14, fill=(235, 235, 235)) -> None:
    ImageDraw.Draw(canvas).text((x, y), text, fill=fill, font=ImageFont.load_default(size=size))


def compare(args) -> None:
    pairs = list(zip(args.runs[::2], args.runs[1::2]))
    side = GRID * PATCH
    head = 40
    canvas = Image.new("RGB", (side * 4, head + len(pairs) * (side + 22) + 30), (20, 20, 20))
    for j, name in enumerate(f"{c} {k}" for c in CAMS for k in ("(model input)", "(kept patches)")):
        labelled(canvas, name, j * side + 6, 8)
    for i, (run, label) in enumerate(pairs):
        images, kept = load(Path(run))
        y = head + i * (side + 22)
        labelled(canvas, f"{label}: {kept.shape[2]} of {GRID * GRID} tokens kept per camera", 6, y, fill=(0, 220, 120))
        canvas.paste(Image.fromarray(row(images, kept, args.call)), (0, y + 20))
    labelled(canvas, "Dropped patches darkened. SigLIP attention has already mixed context into the kept tokens, "
             "so this shows where tokens are kept, not what the model sees.", 6, canvas.height - 22, size=11,
             fill=(170, 170, 170))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)


def gif(args) -> None:
    images, kept = load(Path(args.run))
    side, head, foot = GRID * PATCH, 40, 26
    idx = list(range(0, len(images), args.every))
    w, h = side * 4, head + side + foot
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(args.fps), "-i", "-", "-vf", "split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=5",
           "-loop", "0", str(args.out)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for call in idx:
        canvas = Image.new("RGB", (w, h), (20, 20, 20))
        for j, name in enumerate(f"{c} {k}" for c in CAMS for k in ("(model input)", "(kept patches)")):
            labelled(canvas, name, j * side + 6, 8)
        canvas.paste(Image.fromarray(row(images, kept, call)), (0, head))
        labelled(canvas, f"{args.label}: {kept.shape[2]} of {GRID * GRID} tokens kept per camera   policy call {call}   "
                 "(dropped patches darkened; attention has spread context into the kept tokens)", 6, h - 20, size=11,
                 fill=(170, 170, 170))
        proc.stdin.write(np.asarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB, {len(idx)} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--call", type=int, default=0)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("runs", nargs="+", help="RUN_DIR LABEL pairs")
    c.set_defaults(fn=compare)
    g = sub.add_parser("gif")
    g.add_argument("run")
    g.add_argument("--label", default="")
    g.add_argument("--every", type=int, default=1)
    g.add_argument("--fps", type=int, default=6)
    g.add_argument("--out", type=Path, required=True)
    g.set_defaults(fn=gif)
    a = ap.parse_args()
    a.fn(a)
