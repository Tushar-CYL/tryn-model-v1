"""Phase 1 — tiny Stage-2 alignment run.

Freeze the (random) encoder + decoder, train ONLY the connector on synthetic
image-caption pairs, and watch the loss fall. This is the miniature of the real
alignment stage and the deliverable for the Phase 1 gate.

Run:  python -m training.align          # uses configs/align_tiny.yaml
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from common.checkpoint import save_checkpoint
from common.config import load_config
from common.decoder import DecoderConfig
from common.logging_utils import get_logger, setup_logging
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking
from image_model.data import iter_batches, make_dataset
from image_model.encoder import EncoderConfig
from image_model.model import ImageVLM

log = get_logger(__name__)


def build_model(cfg: DictConfig, vocab_size: int) -> ImageVLM:
    enc_cfg = EncoderConfig(
        image_size=cfg.image.image_size,
        patch_size=cfg.image.patch_size,
        in_channels=cfg.image.in_channels,
        d_model=cfg.encoder.d_model,
        depth=cfg.encoder.depth,
        n_heads=cfg.encoder.n_heads,
        mlp_ratio=cfg.encoder.mlp_ratio,
    )
    dec_cfg = DecoderConfig(
        vocab_size=vocab_size,
        d_model=cfg.decoder.d_model,
        depth=cfg.decoder.depth,
        n_heads=cfg.decoder.n_heads,
        mlp_ratio=cfg.decoder.mlp_ratio,
        max_seq_len=cfg.decoder.max_seq_len,
    )
    return ImageVLM(enc_cfg, dec_cfg, connector_hidden=cfg.connector.hidden_dim)


def run_alignment(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    ds = make_dataset(
        n_samples=cfg.data.n_samples,
        n_classes=cfg.data.n_classes,
        image_size=cfg.image.image_size,
        in_channels=cfg.image.in_channels,
        seed=cfg.seed,
    )
    model = build_model(cfg, vocab_size=ds["tokenizer"].vocab_size).to(device)
    model.freeze_encoder_decoder()  # Stage 2: connector only
    opt = torch.optim.Adam(model.trainable_parameters(), lr=cfg.optim.lr)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    log.info("trainable params: %d / %d (connector only)", n_trainable, n_total)

    first_loss = last_loss = None
    step = 0
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
            if first_loss is None:
                first_loss = last_loss
            if step % 50 == 0:
                log.info("step %d | loss %.4f", step, last_loss)
            run.log({"train/loss": last_loss}, step=step)
            step += 1
            if step >= cfg.optim.steps:
                break

    ckpt = save_checkpoint(
        Path(cfg.output_dir) / cfg.experiment / "align_tiny.pt",
        model,
        step=step,
        first_loss=first_loss,
        final_loss=last_loss,
    )
    run.finish()
    log.info("first_loss=%.4f last_loss=%.4f ckpt=%s", first_loss, last_loss, ckpt)
    return {"first_loss": first_loss, "last_loss": last_loss, "checkpoint": str(ckpt)}


def main() -> None:
    run_alignment(load_config("align_tiny"))


if __name__ == "__main__":
    main()
