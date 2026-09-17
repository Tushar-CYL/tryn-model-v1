"""Phase 2 — Stage 3: instruction tuning with LoRA (Week 6, Days 2-5).

Trains the connector + LoRA adapters on the decoder (encoder + decoder base
frozen) over (image, question, answer) turns with answer-only supervision, then
reports held-out VQA accuracy vs. the majority-answer baseline.

    python -m training.instruct
"""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from common.checkpoint import load_checkpoint, save_checkpoint
from common.config import load_config
from common.logging_utils import get_logger, setup_logging
from common.lora import apply_lora, lora_parameters
from common.seed import resolve_device, set_seed
from common.tracking import init_tracking
from eval.baseline import compare_to_baseline
from eval.vqa import vqa_accuracy
from image_model.instruct_data import iter_vqa_batches, make_vqa_dataset
from image_model.model import ImageVLM

log = get_logger(__name__)


def build_model(cfg: DictConfig, vocab_size: int) -> ImageVLM:
    mcfg = OmegaConf.to_container(cfg, resolve=True)
    mcfg["decoder"]["vocab_size"] = vocab_size
    return ImageVLM.from_config(mcfg)


def run_instruct(cfg: DictConfig) -> dict:
    setup_logging(cfg.logging.level)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    run = init_tracking(cfg)

    train_ds = make_vqa_dataset(cfg.data.n_samples, image_size=cfg.image.image_size,
                                seed=cfg.seed)
    val_ds = make_vqa_dataset(max(cfg.data.n_samples // 4, 32),
                              image_size=cfg.image.image_size, seed=cfg.seed + 999)
    tok = train_ds["tokenizer"]

    model = build_model(cfg, vocab_size=tok.vocab_size).to(device)
    if cfg.get("init_from"):
        load_checkpoint(cfg.init_from, model, map_location=str(device))
        log.info("warm-started from %s", cfg.init_from)

    # Stage 3: freeze base, then attach LoRA (fresh trainable adapters) + connector.
    model.freeze_encoder_decoder()
    n_lora_layers = apply_lora(
        model.decoder, r=cfg.lora.r, alpha=cfg.lora.alpha,
        targets=tuple(cfg.lora.targets), dropout=cfg.lora.dropout,
    )
    model.to(device)
    params = list(model.connector.parameters()) + list(lora_parameters(model.decoder))
    opt = torch.optim.Adam(params, lr=cfg.optim.lr)

    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info("LoRA layers: %d | trainable (connector+LoRA): %d / %d",
             n_lora_layers, n_train, n_total)

    base_before = compare_to_baseline(model, val_ds, device=device)["baseline_accuracy"]
    vqa_before = vqa_accuracy(model, val_ds, device=device).accuracy

    first_loss = last_loss = None
    step = 0
    while step < cfg.optim.steps:
        for batch in iter_vqa_batches(train_ds, cfg.data.batch_size, seed=cfg.seed + step):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            model.train()
            opt.zero_grad()
            loss = model.loss(images, input_ids, text_labels=labels)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            first_loss = first_loss if first_loss is not None else last_loss
            if step % 100 == 0:
                log.info("step %d | loss %.4f", step, last_loss)
            run.log({"train/loss": last_loss}, step=step)
            step += 1
            if step >= cfg.optim.steps:
                break

    cmp = compare_to_baseline(model, val_ds, device=device)
    run.log({"eval/vqa_accuracy": cmp["model_accuracy"],
             "eval/baseline_accuracy": cmp["baseline_accuracy"]}, step=step)
    ckpt = save_checkpoint(Path(cfg.checkpoint.dir) / "instruct.pt", model,
                           step=step, vqa=cmp)
    run.finish()

    log.info("VQA acc %.3f -> %.3f | baseline %.3f | beats=%s",
             vqa_before, cmp["model_accuracy"], base_before, cmp["beats_baseline"])
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "vqa_before": vqa_before,
        "vqa_after": cmp["model_accuracy"],
        "baseline_accuracy": cmp["baseline_accuracy"],
        "beats_baseline": cmp["beats_baseline"],
        "checkpoint": str(ckpt),
        "samples": cmp["model_samples"],
    }


def main() -> None:
    run_instruct(load_config("instruct"))


if __name__ == "__main__":
    main()
