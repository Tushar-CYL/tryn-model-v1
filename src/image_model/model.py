"""Assembled image VLM: [vision tokens | text tokens] -> decoder.

Sequence assembly mirrors the master-plan data flow (§4.1):
    image -> encoder -> patch feats -> connector -> vision tokens
    text  -> decoder.embed_tokens                -> text tokens
    concat -> decoder -> next-token logits

Week 4 makes the encoder and connector pluggable (tiny/SigLIP; MLP/resampler)
while keeping the original constructor working for the Phase-1 tiny run.
"""
from __future__ import annotations

import torch
from torch import nn

from common.connector import MLPConnector, build_connector
from common.decoder import DecoderConfig, TinyDecoder, lm_loss
from .encoder import EncoderConfig, TinyPatchEncoder, build_encoder


class ImageVLM(nn.Module):
    def __init__(
        self,
        enc_cfg: EncoderConfig | None = None,
        dec_cfg: DecoderConfig | None = None,
        connector_hidden: int | None = None,
        *,
        encoder: nn.Module | None = None,
        connector: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ) -> None:
        """Two ways to build:

        * Phase-1 style (kept for compat):
          ``ImageVLM(enc_cfg, dec_cfg, connector_hidden=128)`` — tiny encoder,
          MLP connector, tiny decoder.
        * Pre-built modules (Week 4 factory): pass `encoder`/`connector`/`decoder`.
        """
        super().__init__()
        if encoder is not None:
            if connector is None or decoder is None:
                raise ValueError("Pass encoder, connector, and decoder together.")
            self.encoder = encoder
            self.connector = connector
            self.decoder = decoder
        else:
            if enc_cfg is None or dec_cfg is None:
                raise ValueError("Provide (enc_cfg, dec_cfg) or pre-built modules.")
            self.encoder = TinyPatchEncoder(enc_cfg)
            self.decoder = TinyDecoder(dec_cfg)
            self.connector = MLPConnector(
                in_dim=enc_cfg.d_model, out_dim=dec_cfg.d_model, hidden_dim=connector_hidden
            )

        # Vision token count == connector output tokens if it resamples,
        # else the encoder's patch count (MLP keeps tokens 1:1).
        out_tokens = getattr(self.connector, "out_tokens", None)
        self.num_vision_tokens = out_tokens if out_tokens is not None \
            else self.encoder.num_patches

    # -- factory ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "ImageVLM":
        """Build from a config with `encoder`, `connector`, `decoder` blocks."""
        encoder = build_encoder(cfg["encoder"])
        dec_cfg = DecoderConfig(
            vocab_size=cfg["decoder"]["vocab_size"],
            d_model=cfg["decoder"]["d_model"],
            depth=cfg["decoder"].get("depth", 2),
            n_heads=cfg["decoder"].get("n_heads", 4),
            mlp_ratio=cfg["decoder"].get("mlp_ratio", 2.0),
            max_seq_len=cfg["decoder"].get("max_seq_len", 128),
        )
        decoder = TinyDecoder(dec_cfg)
        conn_cfg = dict(cfg["connector"])
        conn_cfg.setdefault("in_dim", encoder.d_model)
        conn_cfg.setdefault("out_dim", dec_cfg.d_model)
        connector = build_connector(conn_cfg)
        return cls(encoder=encoder, connector=connector, decoder=decoder)

    # -- component freezing (Stage-2 alignment trains the connector only) --------
    def freeze_encoder_decoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # -- core ---------------------------------------------------------------------
    def vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.connector(self.encoder(images))

    def _assemble(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        vision = self.vision_tokens(images)                 # (B, Nv, D)
        text = self.decoder.embed_tokens(text_ids)          # (B, Nt, D)
        return torch.cat([vision, text], dim=1)             # (B, Nv+Nt, D)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """Return logits over the full [vision | text] sequence."""
        return self.decoder(inputs_embeds=self._assemble(images, text_ids))

    def build_labels(
        self, text_ids: torch.Tensor, text_labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Full-sequence labels aligned to [vision | text], vision masked (-100).

        Default (captioning) scores all caption tokens except the leading BOS;
        pass `text_labels` (prompt positions set to -100) for answer-only chat
        supervision.
        """
        nv = self.num_vision_tokens
        b, nt = text_ids.shape
        labels = torch.full((b, nv + nt), -100, dtype=torch.long, device=text_ids.device)
        if text_labels is None:
            text_labels = text_ids.clone()
            text_labels[:, 0] = -100  # don't score predicting BOS
        labels[:, nv:] = text_labels
        return labels

    def loss(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        text_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Next-token LM loss (see `build_labels` for the masking policy)."""
        logits = self.forward(images, text_ids)
        labels = self.build_labels(text_ids, text_labels)
        return lm_loss(logits, labels)
