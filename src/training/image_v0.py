"""Phase 2 — Image model v0 trainer (Week 4, Days 4-5).

Builds the assembled model from `configs/image_v0.yaml` (pluggable encoder +
connector), runs a Stage-2 alignment-style loop (freeze encoder+decoder, train
the connector), logs to W&B, and checkpoints model+optimizer+step so a run can
resume exactly. Uses the real Week-3 shards when present, else a synthetic
fallback so it runs anywhere.

    python -m training.image_v0                       # fresh run
    python -m training.image_v0 checkpoint.resume=outputs/image_v0/ckpt.pt
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from common.checkpoint import load_training_state, save_training_state
from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.seed import resolve_device, set_seed
from common.tokenizer import TinyTokenizer
from common.tracking import init_tracking
from eval.evaluate import evaluate
from image_model.data import iter_batches, load_shard_dataset, make_dataset
from image_model.model import ImageVLM

log = get_logger(__name__)


def _vision_budget(cfg: DictConfig) -> int:
    if cfg.connector.type == "resampler":
        return int(cfg.connector.num_latents)
    return (cfg.image.image_size // cfg.image.patch_size) ** 2


def _load_split(cfg: DictConfig, split: str, seed: int) -> dict | None:
    """Load one split from real shards, or a synthetic stand-in (None if absent)."""
    shards = Path(cfg.data.shards_dir)
    if (shards / split / "manifest.json").exists():
        max_text_len = cfg.decoder.max_seq_len - _vision_budget(cfg) - 1
        normalize = "siglip" if cfg.encoder.type == "siglip" else None
        try:
            return load_shard_dataset(
                shards, image_size=cfg.image.image_size, split=split,
                max_text_len=max_text_len, normalize=normalize,
            )
        except FileNotFoundError:
            return None
    if split == "train":
        return make_dataset(
            n_samples=cfg.data.n_samples, n_classes=cfg.data.n_classes,
            image_size=cfg.image.image_size, in_channels=cfg.image.in_channels, seed=seed,
        )
    # Synthetic held-out: same classes/captions, fresh noise (a fair generalisation test).
    return make_dataset(
        n_samples=max(cfg.data.n_classes * 4, 16), n_classes=cfg.data.n_classes,
        image_size=cfg.image.image_size, in_channels=cfg.image.in_channels, seed=seed + 999,
    )


def _load_data(cfg: DictConfig) -> dict:
    shards = Path(cfg.data.shards_dir)
    if (shards / "train" / "manifest.json").exists():
        log.info("loading real shards from %s", shards)
    else:
        log.info("shards not found at %s -> synthetic fallback", shards)
    return _load_split(cfg, "train", cfg.seed)


def build_model(cfg: DictConfig, vocab_size: int) -> ImageVLM:
    mcfg = OmegaConf.to_container(cfg, resolve=True)
    mcfg["decoder"]["vocab_size"] = vocab_size
    return ImageVLM.from_config(mcfg)


def run_image_v0(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    ds = _load_data(cfg)
    tok: TinyTokenizer = ds["tokenizer"]
    eval_ds = _load_split(cfg, "val", cfg.seed) if cfg.eval.every else None
    model = build_model(cfg, vocab_size=tok.vocab_size).to(device)
    model.freeze_encoder_decoder()  # Stage 2: connector only
    opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg.optim.lr)

    start_step = 0
    if cfg.checkpoint.get("resume"):
        payload = load_training_state(cfg.checkpoint.resume, model, opt, map_location=str(device))
        start_step = int(payload.get("step", 0))
        log.info("resumed from %s at step %d", cfg.checkpoint.resume, start_step)

    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    log.info("vision tokens: %d | trainable params: %d / %d",
             model.num_vision_tokens, n_train, n_total)

    ckpt_dir = Path(cfg.checkpoint.dir)
    ckpt_path = ckpt_dir / "ckpt.pt"
    best_path = ckpt_dir / "best.pt"
    higher_better = cfg.eval.select_by == "token_acc"
    best_metric = None
    best_eval = None

    def _maybe_eval(step: int) -> None:
        nonlocal best_metric, best_eval
        if eval_ds is None or not cfg.eval.every:
            return
        rep = evaluate(model, eval_ds, batch_size=cfg.eval.batch_size)
        run.log({f"eval/{k}": v for k, v in rep.as_dict().items() if k != "n"}, step=step)
        log.info("eval @ %d | loss %.4f | tok_acc %.3f | em %.3f",
                 step, rep.loss, rep.token_acc, rep.exact_match)
        metric = rep.token_acc if higher_better else -rep.loss
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_eval = rep.as_dict()
            save_training_state(best_path, model, opt, step=step, eval=rep.as_dict())

    first_loss = last_loss = None
    step = start_step
    while step < cfg.optim.steps:
        for batch in iter_batches(ds, cfg.data.batch_size, seed=cfg.seed + step):
            images = batch["images"].to(device)
            text_ids = batch["text_ids"].to(device)
            model.train()
            opt.zero_grad()
            loss = model.loss(images, text_ids)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            first_loss = first_loss if first_loss is not None else last_loss
            if step % 50 == 0:
                log.info("step %d | loss %.4f", step, last_loss)
            run.log({"train/loss": last_loss}, step=step)
            step += 1
            if cfg.eval.every and step % cfg.eval.every == 0:
                _maybe_eval(step)
            if step >= cfg.optim.steps:
                break

    _maybe_eval(step)  # final eval
    save_training_state(ckpt_path, model, opt, step=step,
                        first_loss=first_loss, final_loss=last_loss)
    run.finish()
    if last_loss is None:
        log.info("no steps run (already at step %d) ckpt=%s", step, ckpt_path)
    else:
        log.info("first_loss=%.4f last_loss=%.4f ckpt=%s | best=%s",
                 first_loss, last_loss, ckpt_path, best_eval)
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_path) if best_eval is not None else None,
        "best_eval": best_eval,
    }


def main() -> None:
    run_image_v0(load_config("image_v0"))


if __name__ == "__main__":
    main()
