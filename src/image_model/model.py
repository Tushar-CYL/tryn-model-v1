"""Assembled image VLM: [vision tokens | text tokens] -> tiny decoder.

Sequence assembly mirrors the master-plan data flow (§4.1):
    image -> encoder -> patch feats -> connector -> vision tokens
    text  -> decoder.embed_tokens                -> text tokens
    concat -> decoder -> next-token logits
"""
from __future__ import annotations

import torch
from torch import nn

from common.connector import MLPConnector
from common.decoder import DecoderConfig, TinyDecoder, lm_loss
from .encoder import EncoderConfig, TinyPatchEncoder


class ImageVLM(nn.Module):
    def __init__(
        self,
        enc_cfg: EncoderConfig,
        dec_cfg: DecoderConfig,
        connector_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TinyPatchEncoder(enc_cfg)
        self.decoder = TinyDecoder(dec_cfg)
        self.connector = MLPConnector(
            in_dim=enc_cfg.d_model, out_dim=dec_cfg.d_model, hidden_dim=connector_hidden
        )
        self.num_vision_tokens = enc_cfg.num_patches

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

    def loss(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """Next-token LM loss scored on the caption tokens only.

        Vision positions and the leading BOS are masked out (-100).
        """
        logits = self.forward(images, text_ids)
        nv = self.num_vision_tokens
        labels = torch.full(
            (text_ids.size(0), nv + text_ids.size(1)),
            -100,
            dtype=torch.long,
            device=text_ids.device,
        )
        text_labels = text_ids.clone()
        text_labels[:, 0] = -100  # don't score predicting BOS
        labels[:, nv:] = text_labels
        return lm_loss(logits, labels)
