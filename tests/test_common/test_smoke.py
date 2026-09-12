"""Phase 0 · Day 3 gate — imports + config load pass."""
import torch

from common.config import load_config
from common.logging_utils import get_logger
from common.seed import resolve_device, set_seed
from common.tokenizer import BOS, EOS, TinyTokenizer
from common.tracking import init_tracking


def test_imports_and_torch():
    assert torch.__version__  # torch is importable (Day 2 blocker resolved)


def test_config_load_resolves_defaults():
    cfg = load_config("toy")
    # keys from base.yaml (via defaults) and toy.yaml (via _self_) both present
    assert cfg.seed == 0
    assert cfg.tracking.mode == "disabled"
    assert cfg.model.in_dim == 32
    assert cfg.optim.steps == 200


def test_tracking_disabled_is_noop():
    cfg = load_config("toy")
    run = init_tracking(cfg)
    run.log({"x": 1.0}, step=0)  # must not raise / must not need network
    run.finish()


def test_seed_and_device():
    set_seed(0)
    assert resolve_device("cpu").type == "cpu"


def test_tokenizer_roundtrip():
    tok = TinyTokenizer()
    ids = tok.encode("red circle")
    assert ids[0] == BOS and ids[-1] == EOS
    assert tok.decode(ids) == "red circle"


def test_logger():
    assert get_logger("t").name == "t"
