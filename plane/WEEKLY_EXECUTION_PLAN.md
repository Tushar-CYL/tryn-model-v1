# Perception SLM — Week-by-Week Execution Plan (Install → Deploy)

Companion to [`MASTER_PLAN.md`](./MASTER_PLAN.md). This is the **day-level** playbook.

## How to read this
- **~20 weeks, 5 working days/week** (Day 1–Day 5). Weekends = buffer/reading.
- Each day lists: **what you do**, the **folder it touches**, and **✅ done-when** (the checkpoint).
- **Solo path** is linear below. If you have a **2nd person**, they run the **AUDIO track (Weeks 5–8)** in parallel — marked 🔀.
- Adjust dates to your calendar; the *order and dependencies* are what matter.
- **Golden rule:** nothing advances to the next phase until that phase's **gate** (end-of-phase ✅) passes.

**Legend:** 🖥️ local machine · ☁️ rented GPU · 📦 data · 🧪 test · 🔀 parallelizable

---

# PHASE 0 — Setup & smoke test (Week 0)

**Goal:** a clean repo + environment where a *toy* model trains end-to-end. Prove the plumbing before touching real data.

### Week 0
- **Day 1 — Machine & accounts.** Install Python 3.11, Git, CUDA toolkit (or plan GPU rental). Create accounts: HuggingFace, Weights & Biases, GitHub, a GPU provider (RunPod/Lambda/Vast). 🖥️
  ✅ `python --version` = 3.11.x; can log in to HF CLI (`huggingface-cli login`).
- **Day 2 — Repo & env.** `git init`; create `.venv` (`python -m venv .venv` or `uv venv`); `pip install -r requirements.txt`; confirm `import torch; torch.cuda.is_available()`. 🖥️
  ✅ torch imports; CUDA True on GPU box (CPU fine locally for now).
- **Day 3 — Project skeleton.** Add `pyproject.toml`, `configs/base.yaml` (Hydra), logging util in `src/common/`, W&B init helper. Wire `pytest` to discover `tests/`. 🖥️
  🧪 `tests/test_common/test_smoke.py` — imports + config load passes.
- **Day 4 — Toy training loop.** In `src/training/`, write a minimal loop that trains a 2-layer MLP on random tensors, logs to W&B, saves a checkpoint. This is the harness, not the model. ☁️
  ✅ one run appears in W&B; checkpoint file saved.
- **Day 5 — Toy end-to-end + CI.** Add a tiny "fake VLM" (random image tensor → linear → token logits) that runs forward + one train step + eval. Add GitHub Actions running `pytest`. 🧪
  **✅ PHASE 0 GATE:** `pytest` green in CI; toy model trains, evals, and checkpoints end-to-end.

---

# PHASE 1 — Learn the architecture (Weeks 1–2)

**Goal:** understand a real small VLM inside-out by reproducing one, so you build (not copy) in Phase 2.

### Week 1 — Study & run an open small VLM
- **Day 1 — Read.** Study SmolVLM + Moondream model cards and code. Diagram their encoder→connector→decoder on paper. 🖥️
  ✅ you can name every tensor shape from image → answer.
- **Day 2 — Run inference.** Load SmolVLM-256M from HF, run it on 10 sample images, read outputs. ☁️
  ✅ captions/answers print; you log latency + memory.
- **Day 3 — Dissect the connector.** Locate the projector/resampler in the code. Print the shapes going in and out. Write notes in `notebooks/`. 🖥️
  ✅ notebook shows vision-feature shape → token shape mapping.
- **Day 4 — Dissect the encoder.** Inspect the SigLIP/ViT encoder: patch size, hidden dim, #tokens per image. 🖥️
  ✅ documented in `notebooks/encoder_anatomy.ipynb`.
- **Day 5 — Dissect the decoder + tokenizer.** How image tokens are concatenated with text tokens; how the chat template works. 🖥️
  ✅ you can hand-build the input sequence for one example.

### Week 2 — Reproduce a tiny alignment run
- **Day 1 — Tiny dataset.** Grab 2–5k image-caption pairs from HF (e.g., a COCO subset), stream them. 📦
  🧪 `tests/test_data_pipeline/test_load_coco_subset.py`.
- **Day 2 — Wire frozen encoder + decoder.** Load open SigLIP + a small open LM (SmolLM2/Qwen2.5-0.5B), both **frozen**. Add a fresh MLP connector (trainable). 🖥️
  ✅ forward pass produces logits of correct shape.
- **Day 3 — Alignment train (stage 2).** Train ONLY the connector on the tiny dataset for a few hundred steps. ☁️
  ✅ loss goes down; W&B curve logged.
- **Day 4 — Eval + inspect.** Generate captions on held-out images. They'll be rough — that's expected. 🧪
  ✅ `src/eval/` produces caption samples + a simple metric (e.g., CIDEr or loss).
- **Day 5 — Write it up.** Document what worked, shapes, hyperparams. This becomes your Phase-2 recipe. 🖥️
  **✅ PHASE 1 GATE:** you have trained a connector yourself and understand every component.

---

# PHASE 2 — IMAGE model v0 (Weeks 3–7)  ← build this FIRST

**Goal:** *your* small image-understanding model that runs offline on your laptop and beats a baseline on a held-out set. (Its encoder is reused by video.)

### Week 3 — Data pipeline for image
- **Day 1 — Pipeline design.** Define stream→clean→shard flow in `data/pipelines/`. Choose datasets (COCO, CC3M subset, LLaVA-Instruct, VQAv2, DocVQA). 📦
- **Day 2 — Streaming loader.** Implement HF streaming + WebDataset sharding to local/S3. 📦
  🧪 `tests/test_data_pipeline/test_stream_shard.py`.
- **Day 3 — Cleaning/filtering.** Dedup, CLIP-score filter, drop broken images, language filter on captions. 📦
  ✅ filtered shard stats logged (kept/dropped counts).
- **Day 4 — Freeze eval split.** Carve a **frozen held-out** set + a small **golden set** (~200 hand-checked hard cases). 🧪
  ✅ eval split hashed + versioned; never trained on.
- **Day 5 — Data card + smoke.** Write a dataset card (sources, licenses, sizes). Run the full pipeline on a small slice. 📦
  ✅ end-to-end pipeline produces training-ready shards.

### Week 4 — Model assembly
- **Day 1 — Encoder decision.** Choose: open SigLIP init (recommended for v0) vs from-scratch. Wire it in `src/image_model/`. 🖥️
- **Day 2 — Connector.** Implement MLP projector + optional resampler in `src/common/`. 🧪 `tests/test_common/test_connector_shapes.py`.
- **Day 3 — Decoder wiring.** Attach small LM decoder; implement sequence assembly `[vision | text]` + chat template. 🖥️
  🧪 `tests/test_image_model/test_forward.py`.
- **Day 4 — Config + logging.** Hydra config for the full model; W&B; checkpoint save/resume. 🖥️
- **Day 5 — Overfit-one-batch test.** Train on a single batch until loss ≈ 0 (classic sanity check). 🧪
  ✅ model *can* memorize one batch → wiring is correct.

### Week 5 — Alignment training (stage 2)  🔀 *audio track can start now*
- **Day 1 — Full alignment run setup.** Frozen encoder+decoder, train connector on the full aligned dataset. ☁️
- **Day 2–3 — Train + monitor.** Multi-hour/day run; watch loss, sample generations hourly. ☁️
  ✅ captions become coherent.
- **Day 4 — Eval checkpoint.** Run on held-out; log metrics + qualitative samples. 🧪
- **Day 5 — Tune.** Adjust LR/batch/steps; pick best alignment checkpoint. ✅ baseline metric recorded.

### Week 6 — Instruction tuning (stage 3)
- **Day 1 — Instruction data mix.** Blend LLaVA-Instruct + VQAv2 + DocVQA + ChartQA; format as chat. 📦
- **Day 2 — Unfreeze with LoRA/QLoRA.** Train connector + LoRA adapters on decoder (and maybe encoder). ☁️
- **Day 3–4 — Train + eval loop.** Iterate; track VQA accuracy + doc-QA on held-out. 🧪 `tests/test_eval/test_vqa_metric.py`.
- **Day 5 — Beat the baseline.** Compare vs SmolVLM-256M on your held-out set. ✅ you match or beat it on ≥1 task.

### Week 7 — Quantize + run offline
- **Day 1 — Export.** Convert best checkpoint to ONNX and/or GGUF in `src/serving/offline/`. 🧪 `tests/test_serving_offline/test_export.py`.
- **Day 2 — Quantize int8/int4.** Apply AWQ/GPTQ or GGUF quant; measure accuracy drop. ✅ drop < acceptable threshold (set it, e.g. <2%).
- **Day 3 — Run on your laptop.** Inference the quantized model **offline, no GPU**. 🖥️ ✅ answers an image question locally.
- **Day 4 — Latency/memory bench.** Record tokens/s, RAM, model size on CPU. 🧪 `tests/test_serving_offline/test_latency.py`.
- **Day 5 — Package v0.** Save model + a tiny CLI demo (`scripts/ask_image.py`). Write results.
  **✅ PHASE 2 GATE:** your image model runs offline on a laptop and beats a baseline on the held-out set.

---

# PHASE 3 — AUDIO model v0 (Weeks 5–8, runs in parallel 🔀)

**Goal:** independent audio-understanding model — transcribe + answer questions about audio, quantized. (If solo, run these weeks *after* Phase 2; if 2 people, in parallel.)

### Week 5 (or your Week 8) — Audio data
- **Day 1 — Datasets.** LibriSpeech + Common Voice (speech) + ESC-50/AudioSet subset (events). 📦
- **Day 2 — Front-end.** Waveform → 16 kHz → log-mel (80 mels) in `src/audio_model/`. 🧪 `tests/test_audio_model/test_frontend.py`.
- **Day 3 — Streaming loader + augmentation.** SpecAugment, noise mix, VAD segmenting. 📦
- **Day 4 — Frozen eval + golden set.** Hold out clean + noisy + accented samples. 🧪
- **Day 5 — Pipeline smoke.** Full audio pipeline on a slice. ✅ mel-spectrogram batches produced.

### Week 6 — Encoder + connector
- **Day 1 — Audio encoder.** Conv stem + transformer (Whisper-encoder style), or init from open Whisper encoder. 🖥️
- **Day 2 — Connector to decoder.** Project audio frames → decoder tokens. 🧪 `tests/test_audio_model/test_forward.py`.
- **Day 3 — Overfit-one-batch.** Sanity check wiring. ✅ memorizes one batch.
- **Day 4–5 — Alignment train.** Train connector on transcription pairs. ☁️ ✅ transcripts emerge.

### Week 7 — Instruction tuning
- **Day 1–2 — Task mix.** Transcription + audio-QA (AudioCaps/Clotho) + sound classification. Train with LoRA. ☁️
- **Day 3–4 — Eval.** WER for speech; accuracy for events; QA quality. 🧪 `tests/test_eval/test_wer.py`.
- **Day 5 — Robustness.** Test on noisy/accented golden set; add data where weak. ✅ WER acceptable on clean.

### Week 8 — Quantize + offline
- **Day 1–2 — Export + quantize** (GGUF/whisper.cpp path). 🧪 `tests/test_serving_offline/`.
- **Day 3 — Run offline** on laptop; bench latency. 🖥️
- **Day 4 — CLI demo** (`scripts/ask_audio.py`). Day 5 — write-up.
  **✅ PHASE 3 GATE:** audio model transcribes + answers, quantized, runs offline.

---

# PHASE 4 — VIDEO model v0 (Weeks 8–12)

**Goal:** reuse the image encoder, add the time dimension, hit a latency target. This is where efficiency work is hardest.

### Week 8 — Video data
- **Day 1 — Datasets.** MSR-VTT, NExT-QA, Something-Something-v2, ActivityNet-Captions. 📦
- **Day 2 — Frame sampler.** Adaptive 1–8 fps + keyframe selection in `src/video_model/`. 🧪 `tests/test_video_model/test_frame_sampler.py`.
- **Day 3 — Decode pipeline.** Efficient video decode → frames → encoder. 📦
- **Day 4 — Eval + golden set.** Held-out video-QA + captioning. 🧪
- **Day 5 — Pipeline smoke.** ✅ clips → sampled frames → encoder features.

### Week 9 — Temporal module (the hard part)
- **Day 1 — Token math.** Compute tokens = frames × patches; set a budget. 🖥️
- **Day 2 — Token reduction.** Implement ToMe merging + Q-Former resampler. 🧪 `tests/test_video_model/test_token_reduction.py`.
- **Day 3 — Temporal transformer.** Add lightweight temporal attention over frame tokens. 🖥️
- **Day 4 — Connect to decoder.** 🧪 `tests/test_video_model/test_forward.py`. Day 5 — overfit-one-batch. ✅ wiring correct.

### Week 10 — Training
- **Day 1–2 — Alignment.** Train connector+temporal on video-caption pairs (reuse frozen image encoder). ☁️
- **Day 3–4 — Instruction tuning.** Video-QA with LoRA. 🧪 metrics logged.
- **Day 5 — Eval.** Accuracy on NExT-QA held-out. ✅ coherent video answers.

### Week 11 — Optimize latency
- **Day 1 — Bench.** Measure end-to-end latency on target hardware. 🧪 `tests/test_video_model/test_latency.py`.
- **Day 2–3 — Cut it.** Fewer frames, more token merging, quantize; re-measure. ✅ hits target (set it, e.g. <X ms/clip).
- **Day 4 — Accuracy recheck** after cuts. Day 5 — pick best trade-off checkpoint.

### Week 12 — Package
- **Day 1–2 — Export + quantize + offline run.** 🧪 `tests/test_serving_offline/`.
- **Day 3 — CLI demo** (`scripts/ask_video.py`). Day 4–5 — write-up.
  **✅ PHASE 4 GATE:** video model answers questions and meets the latency target.

---

# PHASE 5 — Serving: online + offline (Weeks 11–15, overlaps)

**Goal:** one checkpoint → two production builds (cloud API + edge SDK).

### Week 13 — Online API
- **Day 1 — FastAPI skeleton** in `src/serving/online/` (endpoints for image/video/audio). 🧪 `tests/test_serving_online/test_api.py`.
- **Day 2 — vLLM/TGI backend** wiring + batching. ☁️
- **Day 3 — Queue** (Redis + Celery) for async heavy jobs. 
- **Day 4 — Dockerize** (`deploy/docker/serve.Dockerfile`). 
- **Day 5 — Load test.** ✅ handles concurrent requests; latency logged.

### Week 14 — Offline / edge builds
- **Day 1 — ONNX Runtime build** cross-platform. Day 2 — **GGUF/llama.cpp** CPU build.
- **Day 3 — Mobile** (Core ML for iOS OR ExecuTorch/TFLite for Android — pick one target first). 
- **Day 4 — Edge SDK wrapper** (simple `load()`/`predict()` API). 🧪 `tests/test_serving_offline/test_sdk.py`.
- **Day 5 — Parity check.** ✅ same output online vs offline (within tolerance).

### Week 15 — CI/CD + registry
- **Day 1 — Model registry** (HF private) push/pull. Day 2 — **eval-gate in CI** (block merge on metric regression). 🧪
- **Day 3 — Build pipeline** GitHub Actions → test → build both images → push.
- **Day 4 — Versioning** ties data+model+eval report together. Day 5 — dry-run a full release.
  **✅ PHASE 5 GATE:** one checkpoint serves both online and edge; CI eval-gate active.

---

# PHASE 6 — Optimize & harden (Weeks 13–18, ongoing overlap)

**Goal:** make models smaller, faster, more reliable; add production safety.

### Week 16 — Efficiency pass
- **Day 1 — Distillation setup.** Big teacher VLM → your small model. ☁️
- **Day 2–3 — Distill + quant-aware training.** Re-quantize with QAT; measure gains. ✅ smaller/faster at same accuracy.
- **Day 4 — Pruning experiments.** Day 5 — graph/operator fusion, static shapes, KV-cache tuning. 🧪 latency re-bench.

### Week 17 — Reliability pass
- **Day 1 — Grounding + abstain training** (reduce hallucination; teach "I don't know"). 
- **Day 2 — Expand golden sets** across edge cases per modality. 🧪
- **Day 3 — Drift telemetry** hooks in serving (log inputs stats, confidence). 
- **Day 4 — Shadow-deploy harness** (run new model beside old, compare, no user impact). 
- **Day 5 — Red-team** weird inputs; log failures → data backlog. ✅ failure modes catalogued.

### Week 18 — Buffer / iterate
- Fix the top issues surfaced in Weeks 16–17. Re-train where data was added. Re-run all gates.
  **✅ PHASE 6 GATE:** measurable efficiency + reliability improvement; shadow-deploy works.

---

# PHASE 7 — Deploy (Weeks 19–20)

**Goal:** ship to real users/devices with safe rollout.

### Week 19 — Cloud deploy
- **Day 1 — Kubernetes manifests** (`deploy/cloud/`), autoscaling GPU pool. ☁️
- **Day 2 — Staging deploy** + smoke tests against staging. 🧪
- **Day 3 — Observability** (metrics, alerts, dashboards). 
- **Day 4 — Shadow → canary** (small % of traffic to new model). 
- **Day 5 — Promote** to production if canary healthy. ✅ live API serving.

### Week 20 — Edge deploy + launch
- **Day 1 — Edge SDK release** (downloadable, versioned) + docs. 
- **Day 2 — Device install guide** (`deploy/edge/`) for target hardware. 
- **Day 3 — End-to-end demo** on a real device, offline. 🖥️ ✅ works with no internet.
- **Day 4 — Docs + quickstart** for users. 
- **Day 5 — Launch checklist**: licenses verified, eval report published, rollback plan ready.
  **✅ PHASE 7 GATE:** cloud API + offline edge SDK both shipped and shadow/canary-verified.

---

## At-a-glance timeline

| Weeks | Phase | Output |
|---|---|---|
| 0 | Setup | Repo + toy end-to-end + CI |
| 1–2 | Learn | Reproduced a tiny VLM, understand every part |
| 3–7 | **Image v0** | Offline laptop model beating a baseline |
| 5–8 🔀 | **Audio v0** | Transcribe + audio-QA, quantized |
| 8–12 | **Video v0** | Video-QA meeting latency target |
| 13–15 | Serving | Online API + edge SDK from one checkpoint |
| 16–18 | Optimize | Distilled, quant-aware, hardened, reliable |
| 19–20 | Deploy | Cloud (canary) + offline edge SDK shipped |

## Standing daily habits (every day, all phases)
- Start: pull latest, activate `.venv`, check yesterday's W&B run.
- Commit small; every code change gets/updates a test in the mirrored `tests/test_*` folder.
- End of day: push, note ✅/blockers in a `LOG.md`, back up checkpoints to S3.
- Never merge if the CI **eval-gate** shows a regression.

## Solo vs team note
- **Solo:** run phases strictly in order (Image → Audio → Video). Total ~20–24 weeks.
- **2 people:** Person A owns Image→Video, Person B owns Audio in parallel (Weeks 5–8). Total ~16–18 weeks.
- **Scope cut if time-boxed:** ship **Image v0 offline (Phase 2)** as the first real milestone/demo before committing to all three.
