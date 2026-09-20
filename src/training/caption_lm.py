"""Phase 2 (proper) — caption training with a pretrained LM decoder + LoRA.

Frozen SigLIP + trained connector + pretrained SmolLM2 (LoRA). Production loop:
  * Stage-2 -> Stage-3 curriculum: `connector_warmup` steps train the connector
    against the frozen LM, then LoRA is enabled (the LLaVA recipe).
  * separate LRs (higher for the connector, lower for LoRA), AdamW + weight decay,
    warmup+cosine schedule, grad clipping, gradient accumulation, fp16 AMP (CUDA).
  * held-out loss + BLEU-4, best-checkpoint select, and full resume (model +
    optimizer + step) so long runs survive the 12h free-tier cap.

    python -m training.caption_lm
    python -m training.caption_lm  (with checkpoint.resume=outputs/caption_lm/last.pt in the config)
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.optim import cosine_warmup
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking
from eval.caption_metrics import corpus_bleu4
from image_model.caption_data import iter_caption_batches, load_caption_dataset, to_model_input
from image_model.vlm_lm import build_lm_vlm

log = get_logger(__name__)


def _amp_ctx(device: torch.device, mode: str):
    if mode == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _is_lora(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name


def _optim_groups(model, connector_lr, lora_lr, wd):
    """Two logical groups (connector, LoRA), each split into decay / no-decay."""
    conn = [p for n, p in model.named_parameters() if n.startswith("connector.")]
    lora = [p for n, p in model.named_parameters() if _is_lora(n)]

    def split(params, lr):
        return [
            {"params": [p for p in params if p.ndim >= 2], "lr": lr, "weight_decay": wd},
            {"params": [p for p in params if p.ndim < 2], "lr": lr, "weight_decay": 0.0},
        ]

    return split(conn, connector_lr) + split(lora, lora_lr)


def _set_lora_trainable(model, on: bool):
    for n, p in model.named_parameters():
        if _is_lora(n):
            p.requires_grad_(on)


@torch.no_grad()
def _evaluate(model, ds, device, batch_size, n_samples):
    model.eval()
    total, nb = 0.0, 0
    for batch in iter_caption_batches(ds, batch_size, shuffle=False):
        total += float(model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device), batch["labels"].to(device)))
        nb += 1
    # BLEU over a capped subset, generated in small chunks to bound memory.
    k = min(len(ds["captions"]), max(n_samples, 48))
    gen = []
    for s in range(0, k, 16):
        imgs = to_model_input(ds["images"][s:s + 16]).to(device)
        gen.extend(model.generate_caption(imgs))
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
                                    split="train", max_len=cfg.data.max_len,
                                    max_samples=cfg.data.get("max_train_samples"))
    val_ds = load_caption_dataset(cfg.data.shards_dir, tok, image_size=cfg.image.image_size,
                                  split="val", max_len=cfg.data.max_len,
                                  max_samples=cfg.data.get("max_val_samples", 256))

    wd = cfg.optim.get("weight_decay", 0.01)
    connector_lr = cfg.optim.get("connector_lr", cfg.optim.lr * 5)
    lora_lr = cfg.optim.lr
    opt = torch.optim.AdamW(_optim_groups(model, connector_lr, lora_lr, wd), betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]

    total_steps = cfg.optim.steps
    lr_fn = cosine_warmup(total_steps, warmup_steps=cfg.optim.get("warmup", int(0.05 * total_steps)))
    accum = max(1, cfg.optim.get("accum_steps", 1))
    clip = cfg.optim.get("grad_clip", 1.0)
    amp_mode = cfg.optim.get("amp", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_mode == "fp16" and device.type == "cuda"))
    connector_warmup = cfg.optim.get("connector_warmup", 0)

    # -- resume ---------------------------------------------------------------
    start_step = 0
    if cfg.checkpoint.get("resume"):
        payload = torch.load(cfg.checkpoint.resume, map_location=str(device), weights_only=False)
        model.load_state_dict(payload["trainable_state"], strict=False)
        if "optim_state" in payload:
            opt.load_state_dict(payload["optim_state"])
        start_step = int(payload.get("step", 0))
        log.info("resumed from %s at step %d", cfg.checkpoint.resume, start_step)

    lora_on = start_step >= connector_warmup
    _set_lora_trainable(model, lora_on)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("vision tokens: %d | trainable %d / %d | device %s | amp %s | connector_warmup %d",
             model.num_vision_tokens, n_train, n_total, device, amp_mode, connector_warmup)

    ckpt_dir = Path(cfg.checkpoint.dir)
    best_bleu, best_path = -1.0, ckpt_dir / "caption_lm_best.pt"
    last_path = ckpt_dir / "last.pt"
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    first_loss = last_loss = None
    step = start_step
    opt.zero_grad(set_to_none=True)

    def _batches():
        while True:
            yield from iter_caption_batches(train_ds, cfg.data.batch_size, seed=cfg.seed + step)

    gen = _batches()
    while step < total_steps:
        if not lora_on and step >= connector_warmup:
            _set_lora_trainable(model, True)      # Stage 2 -> Stage 3
            lora_on = True
            log.info("enabled LoRA at step %d (connector warmup done)", step)

        mult = lr_fn(step)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * mult

        micro = 0.0
        for _ in range(accum):
            batch = next(gen)
            model.train()
            with _amp_ctx(device, amp_mode):
                loss = model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device), batch["labels"].to(device)) / accum
            scaler.scale(loss).backward()
            micro += float(loss.detach()) * accum
        if clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        last_loss = micro / accum
        if not (last_loss == last_loss and abs(last_loss) != float("inf")):  # NaN/inf
            raise RuntimeError(
                f"Non-finite loss ({last_loss}) at step {step}. This is almost always "
                f"fp16 overflow (SigLIP on T4) — set optim.amp: none (fp32). "
                f"If it persists on fp32, lower optim.connector_lr / optim.lr."
            )
        first_loss = first_loss if first_loss is not None else last_loss
        if step % 50 == 0:
            log.info("step %d | loss %.4f | lr %.2e | lora %s", step, last_loss,
                     opt.param_groups[0]["lr"], lora_on)
        run.log({"train/loss": last_loss, "train/lr": opt.param_groups[0]["lr"]}, step=step)

        step += 1
        if cfg.eval.every and step % cfg.eval.every == 0:
            metrics, samples = _evaluate(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
            log.info("eval @ %d | val_loss %.4f | BLEU4 %.2f | e.g. %r -> %r",
                     step, metrics["loss"], metrics["bleu4"], samples[0]["target"], samples[0]["generated"])
            run.log({"eval/loss": metrics["loss"], "eval/bleu4": metrics["bleu4"]}, step=step)
            model.save_trainable(last_path, cfg_container, step=step, eval=metrics,
                                 optim_state=opt.state_dict())        # resume point
            if metrics["bleu4"] >= best_bleu:
                best_bleu = metrics["bleu4"]
                model.save_trainable(best_path, cfg_container, step=step, eval=metrics)

    metrics, samples = _evaluate(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
    if metrics["bleu4"] >= best_bleu:
        model.save_trainable(best_path, cfg_container, step=step, eval=metrics)
    ckpt = model.save_trainable(ckpt_dir / "caption_lm.pt", cfg_container, step=step, eval=metrics)
    model.save_trainable(last_path, cfg_container, step=step, eval=metrics, optim_state=opt.state_dict())
    run.finish()
    log.info("done | first %.4f -> last %.4f | val_loss %.4f | BLEU4 %.2f | best_BLEU %.2f",
             first_loss, last_loss, metrics["loss"], metrics["bleu4"], max(best_bleu, metrics["bleu4"]))
    for s in samples:
        log.info("  target: %r | generated: %r", s["target"], s["generated"])
    return {"first_loss": first_loss, "last_loss": last_loss, "val_loss": metrics["loss"],
            "bleu4": metrics["bleu4"], "checkpoint": str(ckpt), "best_checkpoint": str(best_path),
            "samples": samples}


def main() -> None:
    run_caption_lm(load_config("caption_lm"))


if __name__ == "__main__":
    main()
