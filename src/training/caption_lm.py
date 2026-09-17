"""Phase 2 (proper) — caption training with a pretrained LM decoder + LoRA.

Frozen SigLIP + trained connector + pretrained SmolLM2 (LoRA). Production-grade
training loop: AdamW + weight decay, warmup+cosine LR, gradient clipping,
gradient accumulation, optional mixed precision (fp16 on CUDA), held-out loss +
BLEU-4, and small trainable-only checkpoints.

    python -m training.caption_lm            # uses configs/caption_lm.yaml
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.optim import build_optimizer, cosine_warmup, set_lr
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking
from eval.caption_metrics import corpus_bleu4
from image_model.caption_data import iter_caption_batches, load_caption_dataset
from image_model.vlm_lm import build_lm_vlm

log = get_logger(__name__)


def _amp_ctx(device: torch.device, mode: str):
    """autocast context for training; no-op on CPU or when disabled."""
    if mode == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def _evaluate(model, ds, device, batch_size, n_samples):
    model.eval()
    total, nb = 0.0, 0
    for batch in iter_caption_batches(ds, batch_size, shuffle=False):
        total += float(model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device), batch["labels"].to(device)))
        nb += 1
    # BLEU over a capped subset (generation is the slow part)
    k = min(len(ds["captions"]), max(n_samples, 64))
    gen = model.generate_caption(ds["images"][:k].to(device))
    bleu = corpus_bleu4(gen, ds["captions"][:k])
    samples = [{"target": ds["captions"][i], "generated": gen[i]} for i in range(min(n_samples, k))]
    return {"loss": round(total / max(nb, 1), 4), "bleu4": bleu}, samples


def run_caption_lm(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    model = build_lm_vlm(OmegaConf.to_container(cfg, resolve=True)).to(device)
    tok = model.tokenizer
    train_ds = load_caption_dataset(cfg.data.shards_dir, tok, image_size=cfg.image.image_size,
                                    split="train", max_len=cfg.data.max_len)
    val_ds = load_caption_dataset(cfg.data.shards_dir, tok, image_size=cfg.image.image_size,
                                  split="val", max_len=cfg.data.max_len)

    opt = build_optimizer(model, lr=cfg.optim.lr, weight_decay=cfg.optim.get("weight_decay", 0.01))
    total_steps = cfg.optim.steps
    lr_fn = cosine_warmup(total_steps, warmup_steps=cfg.optim.get("warmup", int(0.05 * total_steps)))
    accum = max(1, cfg.optim.get("accum_steps", 1))
    clip = cfg.optim.get("grad_clip", 1.0)
    amp_mode = cfg.optim.get("amp", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_mode == "fp16" and device.type == "cuda"))

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("vision tokens: %d | trainable (connector+LoRA): %d / %d | device %s | amp %s",
             model.num_vision_tokens, n_train, n_total, device, amp_mode)

    ckpt_dir = Path(cfg.checkpoint.dir)
    best_bleu, best_path = -1.0, ckpt_dir / "caption_lm_best.pt"
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    first_loss = last_loss = None
    step = 0
    opt.zero_grad(set_to_none=True)

    def _batches():
        while True:
            yield from iter_caption_batches(train_ds, cfg.data.batch_size, seed=cfg.seed + step)

    gen = _batches()
    while step < total_steps:
        set_lr(opt, cfg.optim.lr * lr_fn(step))
        micro_loss = 0.0
        for _ in range(accum):
            batch = next(gen)
            model.train()
            with _amp_ctx(device, amp_mode):
                loss = model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device), batch["labels"].to(device))
                loss = loss / accum
            scaler.scale(loss).backward()
            micro_loss += float(loss.detach()) * accum
        if clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        last_loss = micro_loss / accum
        first_loss = first_loss if first_loss is not None else last_loss
        if step % 50 == 0:
            log.info("step %d | loss %.4f | lr %.2e", step, last_loss, opt.param_groups[0]["lr"])
        run.log({"train/loss": last_loss, "train/lr": opt.param_groups[0]["lr"]}, step=step)

        step += 1
        if cfg.eval.every and step % cfg.eval.every == 0:
            metrics, samples = _evaluate(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
            log.info("eval @ %d | val_loss %.4f | BLEU4 %.2f | e.g. %r -> %r",
                     step, metrics["loss"], metrics["bleu4"],
                     samples[0]["target"], samples[0]["generated"])
            run.log({"eval/loss": metrics["loss"], "eval/bleu4": metrics["bleu4"]}, step=step)
            if metrics["bleu4"] >= best_bleu:
                best_bleu = metrics["bleu4"]
                model.save_trainable(best_path, cfg_container, step=step, eval=metrics)

    metrics, samples = _evaluate(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
    if metrics["bleu4"] >= best_bleu:
        model.save_trainable(best_path, cfg_container, step=step, eval=metrics)
    ckpt = model.save_trainable(ckpt_dir / "caption_lm.pt", cfg_container, step=step, eval=metrics)
    run.finish()
    log.info("done | first %.4f -> last %.4f | val_loss %.4f | BLEU4 %.2f | ckpt %s | best %s",
             first_loss, last_loss, metrics["loss"], metrics["bleu4"], ckpt, best_path)
    for s in samples:
        log.info("  target: %r | generated: %r", s["target"], s["generated"])
    return {"first_loss": first_loss, "last_loss": last_loss, "val_loss": metrics["loss"],
            "bleu4": metrics["bleu4"], "checkpoint": str(ckpt), "best_checkpoint": str(best_path),
            "samples": samples}


def main() -> None:
    run_caption_lm(load_config("caption_lm"))


if __name__ == "__main__":
    main()
