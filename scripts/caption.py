"""Caption an image with a trained LM captioner (Phase 2, proper).

Rebuilds SigLIP + connector + SmolLM2 from the checkpoint's config (base models
re-download from HF), loads the trained connector + LoRA, and generates.

    python scripts/caption.py --checkpoint outputs/caption_lm/caption_lm_best.pt --image pic.jpg
"""
from __future__ import annotations

import argparse

import torch

from common.seed import resolve_device
from image_model.vlm_lm import LMImageVLM


def _load_image(path, image_size=224):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float() / 255.0
    t = t.view(image_size, image_size, 3).permute(2, 0, 1).contiguous()
    return (t * 2.0 - 1.0).unsqueeze(0)          # SigLIP [-1,1], batch dim


def main() -> None:
    p = argparse.ArgumentParser(description="Caption an image.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=30)
    args = p.parse_args()

    device = resolve_device(args.device)
    model = LMImageVLM.load_trained(args.checkpoint, device=device)
    image_size = 224
    caption = model.generate_caption(_load_image(args.image, image_size),
                                     max_new_tokens=args.max_new_tokens)[0]
    print("caption:", caption)


if __name__ == "__main__":
    main()
