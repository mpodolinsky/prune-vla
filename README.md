# prune-vla

Visual-token pruning for the **pi0.5** vision-language-action policy, evaluated on **LIBERO-10**.

The pruning rule is [DivPrune](https://arxiv.org/abs/2503.02175) (Alvar et al., CVPR 2025): keep the K visual tokens
that are most different from each other (a max-min diversity problem on cosine distance) and drop the rest. It needs
no training and no attention scores. Here it is applied to pi0.5's image tokens, so the Gemma backbone and the action
expert's KV cache see a shorter visual prompt.

```
images -> SigLIP -> projector -> [ prune each real camera: 256 -> K tokens ] -> prefix = [cams | text] -> Gemma + action expert
                                   ^ this repo
```

Nothing in the model is modified. `prune_vla.apply_pruning(policy, PruneConfig(...))` wraps two methods of a LeRobot
`PI05Policy` on the instance and can be undone with `handle.restore()`.

## Methods

| `--method` | what it keeps |
|---|---|
| `none` | all 256 tokens per camera |
| `divprune` | K max-min-diverse tokens (greedy, cosine distance) |
| `random` | K uniformly random tokens (same seed for every call) |
| `pool` | K tokens from average-pooling the 16x16 token grid (K = 64, 16, ...) |

`--compact` additionally drops prefix slots that are padding for the whole batch (pi0.5's unused third camera slot and
the unused language slots). This does not change the model's output and gives the fair "no pruning" baseline for latency.

## Setup

```bash
git clone --recurse-submodules https://github.com/mpodolinsky/prune-vla && cd prune-vla
uv sync --group dev
```

Python 3.12, one CUDA GPU (about 16 GB for the fp32 checkpoint). The LIBERO code is the
[`huggingface/LIBERO`](https://github.com/huggingface/LIBERO) fork in `submodules/LIBERO`; its simulation assets are
downloaded from the Hub (`lerobot/libero-assets`) on first use, and the policy is `lerobot/pi05_libero`.

## Run

```bash
# one condition: 10 LIBERO-10 tasks x 5 episodes, with per-call timings
uv run python -m prune_vla.eval_libero --method divprune --tokens-per-camera 64 --compact --n-episodes 5 --profile

# the full sweep, then a table
N_EPISODES=5 scripts/run_sweep.sh
uv run python scripts/summarize.py
```

Evaluation goes through LeRobot's own `lerobot-eval` loop (image flips, state layout, 520-step limit), so the
`none` row is the stock policy. Extra `lerobot-eval` flags can be appended after `--`. Each run writes
`eval_info.json`, `prune_stats.json` (prefix length, prefill / denoise / total ms per policy call) and rollout videos
to `outputs/<method>_k<K>[_compact]/`.

## Tests

```bash
uv run pytest                 # selection logic (CPU)
uv run pytest -m gpu          # on the real checkpoint: K=256 is bit-identical to the unpatched policy,
                              # compaction leaves actions unchanged, pruned prefix has the expected length
```

## Notes

- Kept tokens get contiguous positions (LeRobot's `cumsum(pad_mask) - 1`); original positions are not preserved.
- pi0.5 was trained on 256 tokens per camera, so pruning is a train/test mismatch; how much it hurts is what the sweep measures.
- The model weights (`lerobot/pi05_libero`, PaliGemma/Gemma terms) are not part of this repository.

## References

- S. Ranjbar Alvar, G. Singh, M. Akbari, Y. Zhang. *DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models.* CVPR 2025.
- Physical Intelligence, *pi0.5*; [LeRobot](https://github.com/huggingface/lerobot); [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
