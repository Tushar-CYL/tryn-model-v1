"""Tiny offline CLI demo (Phase 2, Week 7, Day 5): ask the image model.

Runs fully on CPU. Loads a config + checkpoint, optionally int8-quantizes, then
either captions an image or answers a question about it.

    # caption a synthetic sample with an alignment checkpoint
    python scripts/ask_image.py --config image_v0 --checkpoint outputs/image_v0/best.pt

    # answer a question with the instruction-tuned (LoRA) checkpoint, quantized
    python scripts/ask_image.py --config instruct --checkpoint outputs/image_instruct/instruct.pt \
        --question "what color is it" --quantize --image path/to/img.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from common.checkpoint import load_checkpoint
from common.config import load_config
from common.lora import apply_lora
from common.prompt import build_prompt_ids
from common.tokenizer import TinyTokenizer
from eval.generate import generate, generate_from_prompt
from image_model.instruct_data import make_vqa_dataset
from image_model.model import ImageVLM
from serving.offline.export import quantize_dynamic_int8


def _load_image(path: str | None, image_size: int) -> torch.Tensor:
    if path is None:
        # fall back to a synthetic sample so the demo runs with no assets
        ds = make_vqa_dataset(1, image_size=image_size, seed=7)
        return ds["images"][0]
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = bytearray(img.tobytes())
    t = torch.frombuffer(arr, dtype=torch.uint8).float() / 255.0
    return t.view(image_size, image_size, 3).permute(2, 0, 1).contiguous()


def main() -> None:
    p = argparse.ArgumentParser(description="Ask the offline image model.")
    p.add_argument("--config", default="image_v0", help="image_v0 | instruct")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", default=None, help="image path (synthetic sample if omitted)")
    p.add_argument("--question", default=None, help="ask a question (VQA); else caption")
    p.add_argument("--quantize", action="store_true", help="int8 dynamic quantization")
    p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()

    cfg = load_config(args.config)
    tok = TinyTokenizer()
    mcfg = OmegaConf.to_container(cfg, resolve=True)
    mcfg["decoder"]["vocab_size"] = tok.vocab_size
    model = ImageVLM.from_config(mcfg)

    # instruction checkpoints carry LoRA adapters — recreate them before loading.
    if "lora" in mcfg:
        model.freeze_encoder_decoder()
        apply_lora(model.decoder, r=cfg.lora.r, alpha=cfg.lora.alpha,
                   targets=tuple(cfg.lora.targets), dropout=cfg.lora.dropout)

    load_checkpoint(args.checkpoint, model)
    if args.quantize:
        model = quantize_dynamic_int8(model)
    model.eval()

    image = _load_image(args.image, cfg.image.image_size)

    if args.question:
        gen = generate_from_prompt(model, image, build_prompt_ids(tok, args.question),
                                   max_new_tokens=args.max_new_tokens)
        print(f"Q: {args.question}")
        print(f"A: {tok.decode(gen)}")
    else:
        out = generate(model, image.unsqueeze(0), max_new_tokens=args.max_new_tokens)
        print(f"caption: {tok.decode(out[0].tolist())}")


if __name__ == "__main__":
    main()
