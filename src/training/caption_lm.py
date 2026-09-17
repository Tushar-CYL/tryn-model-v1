"""Phase 2 (proper) — caption training with a pretrained LM decoder + LoRA.

Frozen SigLIP + trained connector + pretrained SmolLM2 (LoRA). Trains on real
COCO caption shards, evaluates by generating captions on held-out images, and
saves only the small trainable state (connector + LoRA adapters).

    python -m training.caption_lm            # uses configs/caption_lm.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking
from image_model.caption_data import iter_caption_batches, load_caption_dataset
from image_model.vlm_lm import LMImageVLM

log = get_logger(__name__)


def build_model(cfg: DictConfig) -> LMImageVLM:
    mcfg = OmegaConf.to_container(cfg, resolve=True)
    model = LMImageVLM.from_pretrained(mcfg)
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.targets), task_type="CAUSAL_LM", bias="none",
    )
    model.lm = get_peft_model(model.lm, lora)
    return model


def _save_trainable(path: Path, model: LMImageVLM, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"trainable_state": state, **extra}, path)
    return path


@torch.no_grad()
def _eval(model, ds, device, batch_size, n_samples):
    model.eval()
    total, nb = 0.0, 0
    for batch in iter_caption_batches(ds, batch_size, shuffle=False):
        total += float(model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                                  batch["attention_mask"].to(device), batch["labels"].to(device)))
        nb += 1
    caps = model.generate_caption(ds["images"][:n_samples].to(device))
    samples = [{"target": ds["captions"][i], "generated": caps[i]} for i in range(len(caps))]
    return {"loss": round(total / max(nb, 1), 4)}, samples


def run_caption_lm(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    model = build_model(cfg).to(device)
    tok = model.tokenizer
    train_ds = load_caption_dataset(cfg.data.shards_dir, tok, image_size=cfg.image.image_size,
                                    split="train", max_len=cfg.data.max_len)
    val_ds = load_caption_dataset(cfg.data.shards_dir, tok, image_size=cfg.image.image_size,
                                  split="val", max_len=cfg.data.max_len)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr)
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("vision tokens: %d | trainable (connector+LoRA): %d / %d",
             model.num_vision_tokens, n_train, n_total)

    ckpt_dir = Path(cfg.checkpoint.dir)
    first_loss = last_loss = None
    step = 0
    while step < cfg.optim.steps:
        for batch in iter_caption_batches(train_ds, cfg.data.batch_size, seed=cfg.seed + step):
            model.train()
            opt.zero_grad()
            loss = model.loss(batch["images"].to(device), batch["input_ids"].to(device),
                              batch["attention_mask"].to(device), batch["labels"].to(device))
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            first_loss = first_loss if first_loss is not None else last_loss
            if step % 50 == 0:
                log.info("step %d | loss %.4f", step, last_loss)
            run.log({"train/loss": last_loss}, step=step)
            step += 1
            if cfg.eval.every and step % cfg.eval.every == 0:
                metrics, samples = _eval(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
                log.info("eval @ %d | val_loss %.4f | e.g. %r -> %r",
                         step, metrics["loss"], samples[0]["target"], samples[0]["generated"])
                run.log({"eval/loss": metrics["loss"]}, step=step)
            if step >= cfg.optim.steps:
                break

    metrics, samples = _eval(model, val_ds, device, cfg.data.batch_size, cfg.eval.n_samples)
    ckpt = _save_trainable(ckpt_dir / "caption_lm.pt", model, step=step, eval=metrics,
                           config=OmegaConf.to_container(cfg, resolve=True))
    run.finish()
    log.info("done | first %.4f -> last %.4f | val_loss %.4f | ckpt %s",
             first_loss, last_loss, metrics["loss"], ckpt)
    for s in samples:
        log.info("  target: %r | generated: %r", s["target"], s["generated"])
    return {"first_loss": first_loss, "last_loss": last_loss, "val_loss": metrics["loss"],
            "checkpoint": str(ckpt), "samples": samples}


def main() -> None:
    run_caption_lm(load_config("caption_lm"))


if __name__ == "__main__":
    main()
