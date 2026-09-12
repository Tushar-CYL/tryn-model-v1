# Perception SLM — Master Plan (Detailed + Diagrams)

**Project:** Small, from-scratch multimodal **understanding** models — Image, Video, Audio.
**Not generation.** Input = pixels / frames / sound → Output = meaning (text, labels, structured JSON, answers).
**Goal:** The most **efficient** small perception models in the world — fast on a $30 chip, private enough for a hospital, cheap enough for a billion calls. On-device first, cloud second. Global.

> **Diagrams render on GitHub and in IDE markdown previewers with Mermaid support.**
> If a diagram shows as raw text, install a Mermaid preview extension, or ask me to publish this as a rendered web page.

---

## Table of contents
1. [The problem](#1-the-problem--what-we-solve)
2. [Pros & cons](#2-pros--cons)
3. [System architecture (big picture)](#3-system-architecture--big-picture)
4. [Each component in depth](#4-each-component-in-depth)
5. [Tech stack — offline / online / cloud](#5-tech-stack--offline--online--cloud)
6. [System design (deployment planes)](#6-system-design--deployment-planes)
7. [Issues & optimization](#7-issues--how-we-optimize)
8. [Datasets (HuggingFace-first)](#8-datasets--huggingface-first)
9. [Roadmap: install → deploy](#9-roadmap--install--deploy)
10. [Repo & test structure](#10-repo--test-structure)

---

## 1. The problem — what we solve

"Understanding" an image/video/audio today = calling a **giant cloud model**. For a real product that means **cost + latency + privacy** pain. Compare the two paths:

```mermaid
flowchart LR
    subgraph CLOUD["❌ Cloud-only (today)"]
        direction TB
        D1["Device / camera / app"] -->|upload raw pixels & audio| API1["Big cloud model<br/>GPT-4o / Gemini / Qwen-72B"]
        API1 -->|answer| D1
        API1 -.-> C1["💸 pay per call"]
        API1 -.-> C2["🐢 300–900 ms round-trip"]
        API1 -.-> C3["🔓 user data leaves device"]
        API1 -.-> C4["📵 needs internet"]
    end
    subgraph EDGE["✅ Our approach (on-device SLM)"]
        direction TB
        D2["Device / camera / app"] --> M2["Small perception model<br/>0.3–3B, quantized, local"]
        M2 -->|answer in ms| D2
        M2 -.-> P1["♻️ ~free inference"]
        M2 -.-> P2["⚡ millisecond latency"]
        M2 -.-> P3["🔒 data never leaves"]
        M2 -.-> P4["🌐 works offline"]
        M2 -->|only heavy jobs| CLOUD2["Optional cloud tier"]
    end
```

**Thesis:** small models (~0.3B–3B params), specialized for perception, quantized, running **on the device** — with an optional cloud tier for heavy/batch work. Small + on-device removes cost, latency, and privacy pain simultaneously.

---

## 2. Pros & cons

```mermaid
mindmap
  root((Perception SLM))
    Pros
      Cost moat: inference ~free vs per-call API
      Privacy: data stays on device (health, finance)
      Latency: ms → real-time video / robotics / AR
      Defensible via DATA + EFFICIENCY not scale
      Field proven: SmolVLM, Moondream, Whisper
      From-scratch realistic at small size
    Cons / Risks
      3 modalities = 3 startups: focus needed
      Data quality/labeling/licensing = 60% of work
      Small models trade accuracy for size
      Train compute real: 8–64 GPU-days per iter
      Big players ship free small models
      Reliability/eval is hard → needs test harness
```

**Bottom line:** you don't win on size. You win on **(a) proprietary task data** and **(b) being the best at running tiny + on-device**. Everything below optimizes for those two.

---

## 3. System architecture — big picture

**Three separate models, one shared core.** Never force one backbone across pixels + frames + sound — it hurts quality. Share the **decoder + tooling**, not the encoders.

```mermaid
flowchart TB
    subgraph CORE["🧠 SHARED PERCEPTION CORE (src/common)"]
        TOK["Tokenizer"]
        DEC["Small language decoder 0.3–1B<br/>(the 'SLM')"]
        CONN["Connector interface<br/>(feature → token projector)"]
        TOOL["Training / Eval / Data tooling"]
    end

    subgraph IMG["🖼️ IMAGE MODEL"]
        IE["Vision encoder<br/>SigLIP/ViT ~80–400M"]
        IC["Connector"]
    end
    subgraph VID["🎬 VIDEO MODEL"]
        VE["Reuses image encoder"]
        VT["Temporal module<br/>frame sample + token merge"]
        VC["Connector (resampler)"]
    end
    subgraph AUD["🔊 AUDIO MODEL"]
        AF["Log-mel front-end"]
        AE["Audio encoder<br/>Whisper-style ~40–300M"]
        AC["Connector"]
    end

    IE --> IC --> DEC
    VE --> VT --> VC --> DEC
    AF --> AE --> AC --> DEC
    CONN -. defines interface .- IC
    CONN -. defines interface .- VC
    CONN -. defines interface .- AC
    DEC --> OUT["Text / labels / JSON answer"]

    style CORE fill:#eef,stroke:#88a
    style IMG fill:#efe,stroke:#8a8
    style VID fill:#ffe,stroke:#aa8
    style AUD fill:#fee,stroke:#a88
```

**Why separate:** Image = spatial only (smallest, fastest to SOTA). Video = image encoder **+ time** (built on top of image). Audio = a totally different encoder (independent, parallelizable).
**What's genuinely "from scratch":** the **connector** (and optionally the decoder). Encoders can be from-scratch *or* init-from-open-weights then re-trained — a budget call (§9).

---

## 4. Each component in depth

### 4.1 Image model — data flow

```mermaid
flowchart LR
    I["Image 384–512px"] --> P["Patchify 14/16px"]
    P --> VIT["ViT / SigLIP encoder<br/>→ grid of patch embeddings"]
    VIT --> PROJ["Connector: MLP projector<br/>patch feats → decoder tokens"]
    PROMPT["Text prompt<br/>'What is on the receipt?'"] --> TOK2["Tokenizer"]
    PROJ --> CAT["Concatenate<br/>[vision tokens | text tokens]"]
    TOK2 --> CAT
    CAT --> DEC2["Decoder 0.3–1B"]
    DEC2 --> ANS["Answer / JSON"]
```
- **Encoder target:** ~80–400M params, patch 14/16, 384–512px input.
- **From scratch:** SigLIP contrastive pretrain on web image-text. **Shortcut:** open SigLIP weights + finetune.
- **First to build** (Phase 2) — its encoder is reused by video.

### 4.2 Video model — the time dimension (where efficiency lives)

```mermaid
flowchart LR
    V["Video clip"] --> FS["Frame sampling 1–8 fps<br/>+ keyframe selection"]
    FS --> ENC["Image encoder (shared)<br/>per-frame patch tokens"]
    ENC --> TM["⚠️ Token explosion here!<br/>Token merge (ToMe) / Q-Former resampler"]
    TM --> TE["Temporal transformer<br/>(motion / events over time)"]
    TE --> VPROJ["Connector"]
    VPROJ --> DEC3["Decoder"]
    DEC3 --> VANS["Video answer / caption / event"]
    style TM fill:#fdd,stroke:#c44
```
- **Core risk:** N frames × M patches = token blow-up → OOM + slow. Fix with frame subsampling, keyframe selection, token merging, resampler.
- Built **on top of** the image encoder → image comes first even under an "audio+video first" preference.

### 4.3 Audio model — independent pipeline

```mermaid
flowchart LR
    W["Waveform 16kHz"] --> VAD["VAD / segment"]
    VAD --> MEL["Log-mel spectrogram<br/>80 mels"]
    MEL --> CONVE["Conv stem + Transformer<br/>(Whisper-encoder style)"]
    CONVE --> APROJ["Connector"]
    APROMPT["Prompt: 'transcribe' /<br/>'what sound?' / 'intent?'"] --> ATOK["Tokenizer"]
    APROJ --> ACAT["Concat [audio tokens | text]"]
    ATOK --> ACAT
    ACAT --> DEC4["Decoder"]
    DEC4 --> AOUT["Transcript / class / intent / QA"]
```
- Understanding, **not just transcription**: sound-event classes, emotion, intent, audio Q&A.
- Fully independent → can be built **in parallel** by a second contributor from day one.

### 4.4 The connector — where OUR data becomes the moat

```mermaid
flowchart TB
    F["Encoder features<br/>(vision grid / audio frames)"] --> OPT{Design choice}
    OPT -->|simple, LLaVA-style| MLP["MLP projector<br/>fast, keeps all tokens"]
    OPT -->|fewer tokens, better for video/audio| QF["Q-Former / Resampler<br/>compresses to K query tokens"]
    MLP --> EMB["Tokens in decoder's<br/>embedding space"]
    QF --> EMB
    EMB --> TRAIN["Trained on OUR (modality,text) pairs<br/>← the defensible asset"]
```

### 4.5 Training stages (curriculum)

```mermaid
flowchart LR
    S1["Stage 1: Encoder pretrain<br/>contrastive (SigLIP) / audio SSL"] --> S2["Stage 2: Alignment<br/>freeze encoder+decoder, train CONNECTOR only"]
    S2 --> S3["Stage 3: Instruction tuning<br/>unfreeze (LoRA/QLoRA), task data"]
    S3 --> S4["Stage 4: Distillation + quant-aware<br/>shrink + int4/int8, keep accuracy"]
    S4 --> S5["Stage 5: Eval gate<br/>golden sets + latency thresholds"]
    style S2 fill:#eef,stroke:#88a
```

### 4.6 The decoder (the "SLM")
- 0.3–1B decoder-only transformer. **From scratch** feasible at this size, or continue-train an open small LM (SmolLM2, Qwen2.5-0.5B/1.5B, Llama-3.2-1B).
- Shared across all three models. Supports **constrained/JSON decoding** for structured extraction.

---

## 5. Tech stack — offline / online / cloud

```mermaid
flowchart TB
    subgraph TRAIN["🏋️ TRAINING (offline plane)"]
        PT["PyTorch 2.x"] --- HF["HF Transformers / Datasets / Accelerate / PEFT"]
        HF --- DS["DeepSpeed / FSDP + FlashAttn2"]
        DS --- WB["W&B tracking"] --- HY["Hydra configs"]
    end
    subgraph ONLINE["☁️ ONLINE serving (cloud tier)"]
        VLLM["vLLM / TGI"] --- FAPI["FastAPI + Uvicorn"]
        FAPI --- RED["Redis + Celery queue"] --- K8["Docker + Kubernetes<br/>GPU autoscale A100/L4"]
    end
    subgraph OFFLINE["📱 OFFLINE / ON-DEVICE"]
        GGUF["GGUF + llama.cpp/whisper.cpp (CPU)"]
        ONNX["ONNX Runtime (cross-platform)"]
        CML["Core ML (iOS/macOS)"]
        ET["ExecuTorch / TFLite (Android)"]
        TRT["TensorRT (Jetson)"]
        Q["Quantization int4/int8: AWQ/GPTQ/GGUF"]
    end
    TRAIN -->|one checkpoint| ONLINE
    TRAIN -->|same checkpoint, quantized build| OFFLINE
```

| Layer | Offline / on-device | Online (cloud) | Cloud infra |
|---|---|---|---|
| Runtime | GGUF, ONNX, Core ML, ExecuTorch, TensorRT | vLLM / TGI | RunPod/Lambda (early) → AWS/GCP |
| Quantize | int4/int8 (AWQ, GPTQ, GGUF) | fp16/int8 | — |
| Storage | on-device file | — | S3 / GCS (data + checkpoints) |
| Data source | — | — | **HuggingFace Hub (streamed)** |
| CI/CD | — | GitHub Actions → test + build + push | model registry (HF private) |

---

## 6. System design — deployment planes

```mermaid
flowchart TB
    CLIENT["📷 Client / device<br/>camera · phone · edge · app"]

    subgraph DEVICE["ON-DEVICE (default path)"]
        RT["Quantized model runtime<br/>ONNX/GGUF/CoreML — ms latency, private"]
    end
    subgraph CLOUDP["CLOUD (fallback / heavy / batch)"]
        GW["API Gateway (FastAPI)<br/>auth · rate-limit"]
        INF["Inference cluster<br/>vLLM/TGI on GPU pool, batched, autoscaled"]
        GW --> INF
    end
    subgraph PLATFORM["SHARED PLATFORM"]
        REG["Model Registry (HF private)"]
        OBJ["Object storage S3<br/>data + checkpoints"]
        TEL["Telemetry / W&B<br/>latency · drift · eval"]
    end
    subgraph TRAINP["TRAINING PLANE (separate, offline)"]
        HFD["HF datasets"] --> DP["Pipeline: clean/label/shard"] --> TR["Train (DeepSpeed)"]
        TR --> EV["Eval"] --> EXP["Quantize / export"]
    end

    CLIENT --> RT
    RT -->|heavy job / low battery| GW
    INF --> REG
    EXP --> REG
    EXP --> OBJ
    REG -->|promote| INF
    REG -->|push edge build| RT
    INF --> TEL
    RT --> TEL
    TEL -->|drift → retrain| DP
```

**Principles:** device-first + cloud-fallback · one checkpoint → two builds · data version + model version + eval report always travel together · shadow-test before promoting.

---

## 7. Issues & how we optimize

```mermaid
flowchart LR
    subgraph EFF["⚡ Efficiency toolbox (ranked)"]
        E1["Quantization int4/8"] --> E2["Distillation (big teacher → small)"] --> E3["Pruning"] --> E4["Token reduction (video)"] --> E5["Graph/operator fusion, static shapes, KV-cache"]
    end
    subgraph REL["🛡️ Reliability toolbox"]
        R1["Golden test sets"] --> R2["CI eval-gate: block merge on regression"] --> R3["Uncertainty / abstain ('I don't know')"] --> R4["Drift monitoring"] --> R5["Shadow deploy before promote"]
    end
```

| Problem | Symptom | Fix |
|---|---|---|
| Video token explosion | OOM, slow | frame subsample, keyframe, token merge (ToMe), resampler |
| Quantization accuracy drop | int4 gets dumb | quant-aware training, mixed 4/8-bit, per-channel, keep sensitive layers high-precision |
| Hallucination | invents unseen text | grounding loss, hard negatives, abstain training, constrained/JSON decode |
| Small model underfits | mediocre everywhere | **distillation** from big teacher, curriculum (easy→hard) |
| Data noise | unstable training | dedup, CLIP-score filter, consensus labels, clean held-out set |
| Field drift | demo works, field fails | collect field data, drift telemetry, periodic retrain |
| Weak-HW latency | too slow on phone | distill smaller, ONNX graph opt, fusion, static shapes |
| Regressions | new model breaks old cases | golden sets + CI eval gate (§10) |
| Audio robustness | fails on noise/accents | SpecAugment, diverse data, VAD front-end |

---

## 8. Datasets — HuggingFace-first ✅

```mermaid
flowchart LR
    HF["HuggingFace Hub"] -->|stream, don't fully download| CLEAN["Clean & filter<br/>dedup · CLIP-score · language"]
    CLEAN --> SHARD["Shard → WebDataset/Parquet on S3"]
    SHARD --> VER["Version: dataset card + hash"]
    VER --> SPLIT["Freeze held-out eval split per version"]
    SPLIT --> TRAINUSE["→ training + eval"]
```

| Modality | Pretrain / align | Instruction / task |
|---|---|---|
| **Image** | LAION subsets, COCO Captions, CC3M/CC12M, SBU, DataComp | LLaVA-Instruct, VQAv2, GQA, TextVQA, DocVQA, OCR-VQA, ChartQA, AI2D, FUNSD |
| **Video** | WebVid, HowTo100M | MSR-VTT, ActivityNet-Captions, NExT-QA, VATEX, Something-Something-v2, Ego4D |
| **Audio** | LibriSpeech, Common Voice, GigaSpeech, VoxPopuli, People's Speech | AudioSet, ESC-50, FSD50K, UrbanSound8K, AudioCaps, Clotho, WavCaps |

> Always verify dataset **licenses** before commercial use.

---

## 9. Roadmap — install → deploy

```mermaid
gantt
    title Perception SLM roadmap (weeks)
    dateFormat  X
    axisFormat  W%s
    section Setup
    Phase 0 env + toy smoke test        :p0, 0, 1
    Phase 1 reproduce tiny VLM          :p1, 1, 2
    section Models
    Phase 2 IMAGE model v0 (first)      :p2, 3, 5
    Phase 3 AUDIO model v0 (parallel)   :p3, 5, 4
    Phase 4 VIDEO model v0 (reuses img) :p4, 8, 5
    section Ship
    Phase 5 serving online+offline      :p5, 11, 5
    Phase 6 optimize + harden           :p6, 13, 6
    Phase 7 deploy cloud + edge SDK     :p7, 16, 4
```

**Phase gates (must pass to advance):**

```mermaid
flowchart LR
    G0["P0: toy model trains+evals end-to-end"] --> G2["P2: image model runs offline on laptop, beats baseline on held-out"]
    G2 --> G3["P3: audio model transcribes + answers, quantized"]
    G3 --> G4["P4: video model hits latency target"]
    G4 --> G5["P5: same checkpoint serves online + edge"]
    G5 --> G7["P7: shadow-tested, promoted, edge SDK shipped"]
```

**Sequencing note (your "audio + video first" preference):** Audio is independent → start **in parallel immediately**. Video needs the image encoder → we build Image first (fast, ~2 wks work) so Video reuses it. You still get audio + video early.

---

## 10. Repo & test structure

Every functional **section** has its own **source folder** and a **mirrored test folder** (`test_<name>`). Tests are first-class — reliability *is* the product.

```mermaid
flowchart TB
    subgraph SRC["src/"]
        C["common"]:::core
        IM["image_model"]
        VM["video_model"]
        AM["audio_model"]
        TRN["training"]
        EVL["eval"]
        SO["serving/online"]
        SF["serving/offline"]
    end
    subgraph TST["tests/  (mirror, proper names)"]
        TC["test_common"]
        TD["test_data_pipeline"]
        TIM["test_image_model"]
        TVM["test_video_model"]
        TAM["test_audio_model"]
        TTR["test_training"]
        TEV["test_eval"]
        TSO["test_serving_online"]
        TSF["test_serving_offline"]
    end
    C --> TC
    IM --> TIM
    VM --> TVM
    AM --> TAM
    TRN --> TTR
    EVL --> TEV
    SO --> TSO
    SF --> TSF
    classDef core fill:#eef,stroke:#88a;
```

**Test tiers (run in CI on every PR):**

```mermaid
flowchart LR
    U["Unit<br/>shapes, forward pass"] --> I["Integration<br/>tiny data → train → eval"] --> GATE["Eval-gate<br/>accuracy + latency thresholds"]
    GATE -->|pass| MERGE["✅ merge allowed"]
    GATE -->|regress| BLOCK["🚫 block merge"]
```

**Directory tree:**
```
img_video_audio/
├── MASTER_PLAN.md · README.md · requirements.txt · .gitignore · pyproject.toml
├── configs/                      # Hydra/YAML per model & experiment
├── data/{raw, processed, pipelines}
├── src/
│   ├── common/                   # tokenizer, connector, decoder, utils
│   ├── image_model/  video_model/  audio_model/
│   ├── training/  eval/
│   └── serving/{online, offline}
├── tests/test_{common,data_pipeline,image_model,video_model,
│               audio_model,training,eval,serving_online,serving_offline}/
├── deploy/{docker, cloud, edge}
├── notebooks/  scripts/
```

---

## Next steps
1. Review these diagrams / depth — tell me what to expand further.
2. I scaffold starter code for `src/common` + `src/image_model` (encoder→connector→decoder skeleton + tests).
3. Phase 0: environment + toy end-to-end smoke test.

*Want this as a rendered, shareable web page (interactive diagrams)? Ask and I'll publish it as an artifact.*
