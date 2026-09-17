"""Image VLM with a PRETRAINED small LM decoder (Phase 2, proper caption quality).

This is the standard "small VLM" recipe — the from-scratch TinyDecoder can't
learn fluent English at CPU/free-GPU scale, so we inherit language ability from a
pretrained small LM (SmolLM2 / Qwen) and train only the connector (+ LoRA on the
LM). The vision encoder (SigLIP) is pretrained and frozen. What's built from
scratch here: the connector and the whole assembly/training/eval pipeline.

    image -> SigLIP (frozen) -> connector (trained) -> vision tokens
    [vision tokens | text embeds] -> pretrained LM (+LoRA) -> next-token logits
"""
from __future__ import annotations

import torch
from torch import nn

from common.connector import build_connector
from .encoder import build_encoder


class LMImageVLM(nn.Module):
    def __init__(self, encoder: nn.Module, connector: nn.Module, lm, tokenizer,
                 num_vision_tokens: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.connector = connector
        self.lm = lm                      # HF AutoModelForCausalLM (maybe PEFT-wrapped)
        self.tokenizer = tokenizer
        self.num_vision_tokens = num_vision_tokens

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, cfg) -> "LMImageVLM":
        """Build from a config dict/DictConfig with `encoder`, `connector`, `lm` blocks."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        encoder = build_encoder(cfg["encoder"])

        lm_name = cfg["lm"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Force fp32: the checkpoint may be bf16, which would clash with the
        # fp32 connector/SigLIP embeddings we feed in as inputs_embeds.
        lm = AutoModelForCausalLM.from_pretrained(lm_name).float()
        if cfg["lm"].get("freeze_base", True):
            for p in lm.parameters():
                p.requires_grad_(False)

        d_lm = lm.config.hidden_size
        conn_cfg = dict(cfg["connector"])
        conn_cfg.setdefault("in_dim", encoder.d_model)
        conn_cfg["out_dim"] = d_lm       # connector must output the LM's embedding dim
        connector = build_connector(conn_cfg)
        n_vis = getattr(connector, "out_tokens", None) or encoder.num_patches
        return cls(encoder, connector, lm, tokenizer, n_vis)

    # -- core ---------------------------------------------------------------------
    def vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.connector(self.encoder(images))          # (B, Nv, D_lm)

    def _embeds_and_mask(self, images, input_ids, attention_mask):
        vis = self.vision_tokens(images)                     # (B, Nv, D)
        tok_emb = self.lm.get_input_embeddings()(input_ids)  # (B, T, D)
        inputs_embeds = torch.cat([vis, tok_emb], dim=1)
        b, nv = vis.shape[0], vis.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        vis_mask = torch.ones(b, nv, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        return inputs_embeds, full_mask

    def forward(self, images, input_ids, attention_mask=None):
        inputs_embeds, full_mask = self._embeds_and_mask(images, input_ids, attention_mask)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask).logits

    def loss(self, images, input_ids, attention_mask, labels):
        """`labels` are the TEXT labels (B, T); vision positions are masked here."""
        inputs_embeds, full_mask = self._embeds_and_mask(images, input_ids, attention_mask)
        b, nv = inputs_embeds.shape[0], self.num_vision_tokens
        vis_labels = torch.full((b, nv), -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([vis_labels, labels], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask,
                       labels=full_labels).loss

    @torch.no_grad()
    def generate_caption(self, images, max_new_tokens: int = 30,
                         repetition_penalty: float = 1.3, no_repeat_ngram_size: int = 3):
        """Greedy caption from vision tokens + a BOS prompt. Returns decoded strings."""
        self.eval()
        device = next(self.lm.parameters()).device
        images = images.to(device)
        b = images.size(0)
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        prompt = torch.full((b, 1), bos, dtype=torch.long, device=device)
        inputs_embeds, full_mask = self._embeds_and_mask(images, prompt, None)
        gen = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
