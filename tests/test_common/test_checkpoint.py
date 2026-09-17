"""load_partial warm-starts only name+shape-matching tensors (no crash on mismatch)."""
import torch
from torch import nn

from common.checkpoint import load_partial, save_checkpoint


def test_load_partial_copies_only_compatible(tmp_path):
    src = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
    path = save_checkpoint(tmp_path / "src.pt", src)

    # Target: first layer matches shape, second differs (4->3 vs 4->2).
    tgt = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
    info = load_partial(path, tgt)

    assert info["loaded"] > 0 and info["skipped"] > 0
    # Matching layer copied (all ones); mismatched layer left as-is.
    assert torch.allclose(tgt[0].weight, torch.ones_like(tgt[0].weight))
    assert tgt[1].weight.shape == (3, 4)


def test_load_partial_all_mismatch_is_safe(tmp_path):
    src = nn.Linear(8, 8)
    path = save_checkpoint(tmp_path / "s.pt", src)
    tgt = nn.Linear(4, 4)  # nothing matches
    info = load_partial(path, tgt)
    assert info["loaded"] == 0  # no crash, just nothing copied
