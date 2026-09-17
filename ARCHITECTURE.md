# Architecture — what's from scratch, what's pretrained

This project builds a small **image-understanding** model (image → caption). It is
built the way real small VLMs (SmolVLM, LLaVA, Moondream) are built: reuse
pretrained perception + language backbones, and **train the bridge between them
from scratch**. Being honest about this boundary matters.

## The pipeline

```
        image (224×224)
             │
      ┌──────▼───────┐   PRETRAINED, FROZEN
      │   SigLIP     │   google/siglip-base-patch16-224 — vision features
      └──────┬───────┘   (196 patch tokens × 768-d)
             │
      ┌──────▼───────┐   ★ FROM SCRATCH — trained by us
      │  Connector   │   PerceiverResampler: 196 tokens → 32 tokens, 768→d_lm
      │ (resampler)  │   the learned "translation" from vision space to LM space
      └──────┬───────┘
             │  [32 vision tokens | caption tokens]
      ┌──────▼───────┐   PRETRAINED + LoRA (small trained adapters)
      │   SmolLM2    │   HuggingFaceTB/SmolLM2-135M/360M — language fluency
      └──────┬───────┘
             │
        next-token logits → caption
```

## From scratch (our engineering)
- **The connector** — a Perceiver-style resampler (learned latent queries,
  cross-attention) that compresses SigLIP's 196 patch tokens to a fixed budget and
  projects them into the LM's embedding space. This is the model's core learned
  asset. (`src/common/connector.py`)
- **LoRA** — our own low-rank adapters, applied to the LM's attention projections.
  (`src/common/lora.py`, no `peft` dependency.)
- **The entire system**: data pipeline (stream → clean → dedup/CLIP-filter → split
  → shard, with a frozen, checksummed eval split), sequence assembly + label
  masking, the training loop (AdamW, warmup+cosine LR, grad clipping, gradient
  accumulation, mixed precision), held-out eval (loss + BLEU-4), quantization +
  offline serving, and the checkpoint/inference APIs.

## Pretrained (inherited, not our claim)
- **SigLIP** vision encoder — frozen; provides visual features. Training it from
  scratch needs web-scale image-text data.
- **SmolLM2** language model — frozen base + our LoRA; provides English fluency.
  Training an LM from scratch to fluency needs billions of tokens / many GPU-days,
  which no free tier can do. (An earlier from-scratch `TinyDecoder` is kept only as
  a Phase-0/1 teaching stub; it cannot produce fluent captions and is not the real
  model.)

## Why this split
A "small VLM from scratch" does **not** mean training a language model from
scratch — it means building the multimodal integration from scratch on top of
open backbones. That connector + pipeline is the real, defensible work, and it is
entirely ours.

## Reproduce
See `README.md` → *Reproduce*. The real captioner is `training/caption_lm.py`
(config `configs/caption_lm.yaml`); run it on a free Kaggle T4 with
`notebooks/train_caption_kaggle.ipynb`.
