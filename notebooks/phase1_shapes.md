# Phase 1 — tensor shapes end-to-end (tiny from-scratch VLM)

Config: `configs/align_tiny.yaml`. All modules are randomly initialised (no downloads).

| Stage | Module | In shape | Out shape |
|---|---|---|---|
| Image in | — | `(B, 3, 32, 32)` | — |
| Patchify + encode | `TinyPatchEncoder` (patch 8 → 4×4=16 patches, d=64) | `(B, 3, 32, 32)` | `(B, 16, 64)` |
| Project to token space | `MLPConnector` (64 → 128 → 64) | `(B, 16, 64)` | `(B, 16, 64)` |
| Text embed | `TinyDecoder.embed_tokens` (vocab=42, d=64) | `(B, T)` ids | `(B, T, 64)` |
| Sequence assembly | `torch.cat([vision, text], dim=1)` | — | `(B, 16+T, 64)` |
| Decode | `TinyDecoder` (causal, depth 2) | `(B, 16+T, 64)` | `(B, 16+T, 42)` logits |
| Loss | `lm_loss` (next-token CE) | logits + labels | scalar |

**Label masking (`ImageVLM.loss`):** vision positions and the leading `BOS`
are set to `-100` (ignored); only real caption tokens + `EOS` are scored.

**Stage-2 alignment:** `freeze_encoder_decoder()` → only the `MLPConnector`
(~parameters) is trained. See `training/align.py`; loss falls ~3.83 → ~3.04 in
300 CPU steps, confirming the connector learns to steer the frozen decoder.

**Overfit-one-batch (wiring check):** with nothing frozen, `model.loss` on a
single 8-sample batch drops below 0.1 within 300 steps — proving the assembly
and masking are correct (`tests/test_image_model/test_overfit_one_batch.py`).
