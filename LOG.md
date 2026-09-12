# Build Log

Daily progress against [`WEEKLY_EXECUTION_PLAN.md`](./plane/WEEKLY_EXECUTION_PLAN.md).

---

## Phase 0 · Week 0

### Day 1 — Machine & accounts
**Machine (verified):**
- ✅ Git 2.54.0
- ✅ uv 0.11.28
- ✅ **Python 3.11.15** installed via `uv python install 3.11` (project will pin this; global default stays 3.14)
- ⚠️ No local NVIDIA GPU → training runs on rented GPU (RunPod/Lambda). Laptop = coding + CPU inference. As planned (☁️).

**Accounts (your action — checklist):**
- [ ] GitHub — repo host + CI
- [ ] HuggingFace — datasets + model registry; create a **Read/Write access token** (Settings → Access Tokens)
- [ ] Weights & Biases — training metrics; copy your **API key** (wandb.ai/authorize)
- [ ] GPU provider — RunPod **or** Lambda **or** Vast.ai (only needed from Week 5; can defer)

**HuggingFace login:** ✅ logged in as `LNTTushar` (org `build-small-hackathon`), token type fine-grained (read OK; add write scope before model-registry push in Wk15).
**wandb login:** deferred to first training run (Day 4).

**✅ DAY 1 GATE MET:** Python 3.11.15 + HF CLI login working.

### Day 2 — Repo & env (in progress)
- ✅ `.venv` created on Python 3.11.15 (`uv venv --python 3.11`)
- ✅ `huggingface_hub` 1.31 + `wandb` 0.30 installed
- ✅ full stack installed (torch 2.14.0+cpu, transformers 5.17, datasets 5.0, accelerate, peft, hydra, pytest, etc.)
- ❌ **`import torch` fails: WinError 4551 — Smart App Control blocks torch DLLs.**

**BLOCKER (Day 2) — RESOLVED:** `import torch` now works on CPU (`torch 2.14.0+cpu`, `cuda False`). Smart App Control no longer blocks the DLLs. All Phase 0/1 work below runs locally on CPU.

### Day 3 — Project skeleton ✅
- ✅ `pyproject.toml` (src layout, `pytest pythonpath=src`, `uv pip install -e .`)
- ✅ Hydra configs: `configs/{base,toy,align_tiny}.yaml` + lightweight `defaults` loader (`common/config.py`)
- ✅ `common/`: `logging_utils`, `seed`, `tokenizer` (tiny char-level, no downloads), `tracking` (W&B helper, network-free by default), `checkpoint`, `connector` (MLP projector), `decoder` (from-scratch TinyDecoder)
- 🧪 `tests/test_common/test_smoke.py` + `test_connector_shapes.py` — green

### Day 4 — Toy training loop ✅
- ✅ `training/toy_loop.py` — 2-layer MLP on random tensors, logs via tracking helper, saves checkpoint. Loss 36.4 → 15.2.
- 🧪 `tests/test_training/test_toy_loop.py` — loss decreases + checkpoint written

### Day 5 — Fake VLM + CI ✅  **← PHASE 0 GATE MET**
- ✅ `image_model/fake_vlm.py` — random image → linear → token logits; forward + train step + eval
- ✅ `.github/workflows/ci.yml` — GitHub Actions runs `pytest` on CPU torch
- 🧪 `tests/test_image_model/test_fake_vlm.py` — green

---

## Phase 1 · Weeks 1–2 (tiny from-scratch reproduction, CPU, no downloads)

Chosen scope: reproduce the encoder→connector→decoder pipeline with **tiny random-init stubs** instead of downloading SmolVLM, then do a real Stage-2 alignment run.

- ✅ `image_model/encoder.py` — `TinyPatchEncoder` (ViT-style, 16 patch tokens, d=64)
- ✅ `image_model/model.py` — `ImageVLM`: sequence assembly `[vision | text]`, label masking, `freeze_encoder_decoder()`
- ✅ `image_model/data.py` — synthetic per-class image/caption pairs
- ✅ `training/align.py` — **Stage-2 alignment**: freeze encoder+decoder, train connector only. Loss 3.83 → 3.04 (300 CPU steps).
- ✅ `eval/generate.py` (greedy) + `eval/metrics.py` (token acc, exact match)
- ✅ `notebooks/phase1_shapes.md` — full tensor-shape map
- 🧪 `test_forward.py`, `test_overfit_one_batch.py` (loss→<0.1, wiring correct), `test_alignment.py` (connector reduces loss), `test_generate.py`

**✅ PHASE 1 GATE MET:** trained a connector from scratch; every component (encoder, connector, decoder, tokenizer, sequence assembly, masking) understood and tested. **17/17 tests green.**

**Next (Phase 2):** swap tiny stubs for open SigLIP + a small open LM; build the real image data pipeline (Week 3); GPU needed from Week 5 (alignment/instruction-tuning runs) — set up RunPod/Lambda before then.
