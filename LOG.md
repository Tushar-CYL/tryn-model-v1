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

---

## Phase 2 · Week 3 — Image data pipeline ✅

New package `src/data_pipeline/` (mirrored by `tests/test_data_pipeline/`). Dataset-agnostic
`Record = {id, image, caption}` flows through stream → clean → split → shard. Real Hub
datasets stream in; a synthetic source keeps the whole pipeline + tests offline on CPU.

- ✅ **Day 1 — Design + dataset registry.** `sources.py` registry: `coco`
  (`clip-benchmark/wds_mscoco_captions`, inline jpg+txt), `coco_karpathy`, `cc3m`.
- ✅ **Day 2 — Streaming + sharding.** `sources.iter_hf` (lazy `datasets` streaming) +
  `shard.py` WebDataset `.tar` writer/reader. 🧪 `test_stream_shard.py` (roundtrip).
- ✅ **Day 3 — Cleaning/filtering.** `clean.py`: broken/tiny images, empty/too-long
  captions, non-English (ASCII-ratio), caption+image dedup, optional CLIP-score hook.
  Logs kept/dropped counts. 🧪 `test_clean.py`.
- ✅ **Day 4 — Freeze eval split.** `splits.py`: hash-based deterministic train/val/golden,
  frozen manifest with sha256, `FrozenEvalSet` leakage guard. 🧪 `test_splits.py`.
- ✅ **Day 5 — Data card + smoke.** `build.py` orchestrator (CLI) → `report.json` +
  `DATA_CARD.md`. **Real run:** streamed 60 COCO pairs → `data/processed/coco_v0`
  (train/val/golden shards); single-line captions, 60 kept.

## Phase 2 · Week 4 — Model assembly ✅

- ✅ **Day 1 — Encoder decision.** `encoder.build_encoder`: `tiny` (offline default) or
  `siglip` (`SiglipVisionEncoder`, open `google/siglip-base-patch16-224`, frozen). Same
  interface. **Real SigLIP forward verified:** (2,3,224,224)→(2,196,768).
- ✅ **Day 2 — Connector + resampler.** `connector.PerceiverResampler` (learned latents,
  cross-attn) fixes the vision-token budget regardless of input length; `build_connector`
  factory. 🧪 `test_connector_shapes.py` (196→16).
- ✅ **Day 3 — Decoder wiring + chat template.** `ImageVLM.from_config` (pluggable
  encoder/connector); `prompt.py` chat template with **answer-only label masking**;
  `ImageVLM.loss(text_labels=...)` for instruction-style supervision. 🧪 `test_prompt.py`,
  `test_model_v0.py`.
- ✅ **Day 4 — Config + logging + resume.** `configs/image_v0.yaml`; `training/image_v0.py`
  trains connector on real shards (or synthetic fallback), `save/load_training_state`
  (model+optimizer+step). **Real run:** loss 3.79→3.31 on COCO shards. 🧪 `test_image_v0.py`.
- ✅ **Day 5 — Overfit-one-batch.** v0 (tiny+resampler) memorises a batch (loss→<0.1).
  🧪 `test_model_v0.py::test_overfit_one_batch_v0`.

**Status:** **33/33 tests green** (17 + 16 new), CPU + offline. Pipeline and model both
verified against real COCO data and real SigLIP weights.

## Phase 2 · Week 5 — Alignment + eval ✅

- ✅ **Day 1–3 — Alignment.** `training/image_v0.py` Stage-2 (connector-only) on the real
  COCO shards: loss **3.79 → 3.31**.
- ✅ **Day 4 — Eval checkpoint.** `eval/evaluate.py`: held-out loss + token-accuracy +
  exact-match + qualitative samples. Fixed `metrics.exact_match` to strip trailing EOS.
  🧪 `test_evaluate.py`.
- ✅ **Day 5 — Tune / best-ckpt.** Trainer runs held-out eval every N steps and saves
  `best.pt` by the selected metric (`eval.select_by`). Baseline metric recorded.

## Phase 2 · Week 6 — Instruction tuning (LoRA) ✅

- ✅ **Day 1 — Instruction data.** `image_model/instruct_data.py`: (image, question, answer)
  VQA turns via the chat template (answer-only masking). Real VQA mixes stream through the
  same path. Fixed `make_dataset`/`make_vqa_dataset` to share per-class bases across
  train/held-out (`base_seed`) so generalisation is real.
- ✅ **Day 2 — LoRA.** `common/lora.py` (`LoRALinear`, `apply_lora`, `merge_lora`) — from
  scratch, freezes base, trains low-rank adapters. 🧪 `test_lora.py`.
- ✅ **Day 3–4 — Train + eval.** `training/instruct.py` trains connector + decoder LoRA;
  `eval/vqa.py` VQA accuracy. 🧪 `test_vqa_metric.py`.
- ✅ **Day 5 — Beat the baseline.** Held-out VQA **0.00 → 1.00** vs majority baseline **0.19**
  → beats it. `eval/baseline.py` (SmolVLM comparison wired as GPU opt-in). 🧪 `test_instruct.py`.

## Phase 2 · Week 7 — Quantize + offline + package ✅  **← PHASE 2 GATE MET**

- ✅ **Day 1 — Export.** `serving/offline/export.py`: ONNX export of the full forward
  (runs under onnxruntime). 🧪 `test_export.py`.
- ✅ **Day 2 — Quantize.** int8 dynamic quant: 843 KB → 583 KB (**1.45×**); token-acc drop
  < 0.15.
- ✅ **Day 3 — Run offline.** `scripts/ask_image.py` answers on CPU, no network (int8).
- ✅ **Day 4 — Bench.** `serving/offline/bench.py`: 203K params · 0.8 MB · 6.2 ms/image ·
  3243 tok/s. 🧪 `test_latency.py`.
- ✅ **Day 5 — Package + results.** CLI demo + `RESULTS.md`.

**✅ PHASE 2 GATE MET:** image model runs offline on a laptop and beats a baseline on the
held-out set. **45/45 tests green.** See `RESULTS.md`.

**Next (Phase 3, Audio — or scale Image on GPU):** the whole stack (data pipeline, model,
alignment, LoRA, eval, quantization, serving) is CPU-verified and scale-ready. Flip
`encoder.type: siglip`, point at a larger pull, set `tracking.mode: online`, run Weeks 5–6
on a GPU to scale Image v0; or start Phase 3 (audio).
