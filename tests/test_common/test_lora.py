"""LoRA: correct shapes, frozen base, trainable adapters, and exact merge."""
import torch
from torch import nn

from common.lora import LoRALinear, apply_lora, lora_parameters, merge_lora


def test_lora_linear_starts_as_identity_and_is_trainable():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)
    # B initialised to 0 -> adapter is a no-op at start (== base output).
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    # base frozen, adapters trainable
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.weight.requires_grad and lora.lora_B.weight.requires_grad


def test_merge_matches_forward():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=8)
    nn.init.normal_(lora.lora_B.weight, std=0.1)  # make the adapter non-trivial
    x = torch.randn(5, 16)
    merged = lora.merge()
    assert torch.allclose(lora(x), merged(x), atol=1e-5)


def test_apply_lora_targets_and_params():
    dec = nn.Sequential()
    dec.add_module("mlp", nn.Linear(8, 8))
    dec.add_module("other", nn.Linear(8, 8))
    n = apply_lora(dec, r=2, targets=("mlp",))
    assert n == 1
    assert isinstance(dec.mlp, LoRALinear)
    assert not isinstance(dec.other, LoRALinear)
    assert len(list(lora_parameters(dec))) == 2  # A and B
    assert merge_lora(dec) == 1
    assert isinstance(dec.mlp, nn.Linear)
