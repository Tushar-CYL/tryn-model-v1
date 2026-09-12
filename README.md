# Perception SLM

Small, from-scratch multimodal **understanding** models — Image, Video, Audio.
Not generation. Input = pixels/frames/sound → Output = meaning (text, labels, structured data).
Efficiency-first, on-device + cloud. Global.

**Read [`MASTER_PLAN.md`](./MASTER_PLAN.md) first** — full problem statement, architecture, tech stack, datasets, phased roadmap, and folder/test conventions.

## Layout
- `src/common` — shared core (tokenizer, connector, small decoder)
- `src/image_model` / `src/video_model` / `src/audio_model` — the three models
- `src/training` · `src/eval` · `src/serving/{online,offline}`
- `tests/test_*` — one test folder per section (unit / integration / eval-gate)
- `data/` · `deploy/` · `configs/` · `scripts/` · `notebooks/`

## Quick start (Phase 0–1)
```bash
uv venv --python 3.11 && .venv/Scripts/activate      # Windows; use bin/activate on *nix
uv pip install -r requirements.txt
uv pip install -e .                                  # makes `common`, `image_model`, ... importable
pytest                                               # 17 tests: harness + tiny VLM (all green)
```

Run the two learning demos (both CPU-only, no downloads, no network):
```bash
python -m training.toy_loop     # Phase 0 · Day 4 — toy MLP trains + checkpoints
python -m training.align        # Phase 1     — connector-only alignment, loss falls
```
Tracking is off by default (`configs/base.yaml → tracking.mode: disabled`); set it
to `online`/`offline` on the GPU box. Tensor shapes: `notebooks/phase1_shapes.md`.
