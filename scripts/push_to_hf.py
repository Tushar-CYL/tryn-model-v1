"""Push a trained checkpoint + config + model card to the HuggingFace Hub.

Run after training is happy (Phase 2 done). Requires a logged-in HF token with
write scope (`huggingface-cli login`, or set HF_TOKEN).

    python scripts/push_to_hf.py \
        --repo-id LNTTushar/perception-slm-image-v0 \
        --checkpoint outputs/image_v0_gpu/best.pt \
        --config image_v0_gpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import CONFIG_DIR


def _model_card(repo_id: str, config: str, eval_info: dict | None) -> str:
    metrics = ""
    if eval_info:
        rows = "\n".join(f"| {k} | {v} |" for k, v in eval_info.items())
        metrics = f"\n## Held-out metrics\n\n| metric | value |\n|---|---|\n{rows}\n"
    return f"""---
library_name: perception-slm
tags:
  - multimodal
  - image-understanding
  - from-scratch
  - vlm
license: apache-2.0
---

# {repo_id}

Small, **from-scratch** image-understanding model (Perception SLM, Phase 2 / image v0).
Encoder → connector (resampler) → tiny decoder, trained with Stage-2 alignment and
Stage-3 LoRA instruction tuning. Config: `{config}`.
{metrics}
## Files
- `model.pt` — checkpoint (`model_state` + training metadata)
- `config.yaml` — the exact config used to build the model
- `model_int8.pt` — int8-quantized weights for CPU/offline (if uploaded)

## Load
```python
from huggingface_hub import hf_hub_download
import torch
ckpt = torch.load(hf_hub_download("{repo_id}", "model.pt"), map_location="cpu")
# rebuild ImageVLM.from_config(config) then load_state_dict(ckpt["model_state"])
```

Built with the [perception-slm](https://github.com/) repo; see its `RESULTS.md`.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Push a checkpoint to the HF Hub.")
    p.add_argument("--repo-id", required=True, help="e.g. LNTTushar/perception-slm-image-v0")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="config name, e.g. image_v0_gpu")
    p.add_argument("--int8", default=None, help="optional path to an int8 checkpoint")
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="HF token (else uses cached login/HF_TOKEN)")
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    # pull any eval metadata out of the checkpoint for the card
    eval_info = None
    try:
        import torch
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        eval_info = payload.get("eval") or payload.get("vqa")
    except Exception:
        pass

    api.upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="model.pt",
                    repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=str(CONFIG_DIR / f"{args.config}.yaml"),
                    path_in_repo="config.yaml", repo_id=args.repo_id)
    if args.int8:
        api.upload_file(path_or_fileobj=args.int8, path_in_repo="model_int8.pt",
                        repo_id=args.repo_id)

    card = _model_card(args.repo_id, args.config,
                       eval_info if isinstance(eval_info, dict) else None)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo_id)

    print(f"Pushed to https://huggingface.co/{args.repo_id}")
    print(f"  model.pt, config.yaml, README.md" + (", model_int8.pt" if args.int8 else ""))
    if eval_info:
        print("  metrics:", json.dumps(eval_info))


if __name__ == "__main__":
    main()
