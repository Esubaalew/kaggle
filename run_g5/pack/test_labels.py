import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import build_labels as B
from data import PairDataset
from grounded import GroundedMorphMemory, grounding_loss


def test_all_root_states_survive_loader():
    tok = Tokenizer(WordLevel({"[UNK]": 3, "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    class Cache:
        entries = {"a": {"root": 1}, "b": {"root": -1}, "c": {"root": 0}, "d": {"root": -2}}
    cfg = {"aux_ignore_index": -100, "max_len": 32, "bos_id": 1, "eos_id": 2}
    item = PairDataset([("a b c d", "a")], tok, Cache(), cfg, "train")[0]
    assert item[2].tolist() == [-100, 1, -1, 0, -2, -100]


def test_gate_gradient_excludes_oob_unknown_and_special():
    mem = GroundedMorphMemory(8, 4, 3, dropout=0)
    _, info = mem(torch.randn(1, 5, 8), word_ids=torch.arange(5).unsqueeze(0))
    info["gate_live"].retain_grad()
    loss, _ = grounding_loss([info], torch.tensor([[1, 0, -1, -2, -100]]), torch.tensor([[1, 0, 2, -100, -100]]))
    assert torch.isfinite(loss)
    loss.backward(); g = info["gate_live"].grad[0]
    assert g[0] < 0 and g[1] > 0
    assert torch.equal(g[2:], torch.zeros_like(g[2:]))


def test_all_unlabelled_batch_has_finite_loss_and_backward():
    mem = GroundedMorphMemory(8, 4, 3, dropout=0)
    _, info = mem(torch.randn(1, 3, 8), word_ids=torch.arange(3).unsqueeze(0))
    lab = torch.full((1, 3), -2)
    loss, _ = grounding_loss([info], lab, lab, entropy_weight=.01)
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.isfinite(mem.mk_root.grad).all()


def test_builder_distinguishes_states_and_keeps_patterns():
    words = ["ቀተለ", "ሰበረ", "ውስተ", "ሀሀ"]
    c = B.build_cache_dict(words, words, {"ቀተለ": ("ቀ-ተ-ለ", "basic_past"), "ሰበረ": ("ሰ-በ-ረ", "gerund"), "ውስተ": (None, "__rootless__"), "ሀሀ": (None, None)}, k_root=1)
    assert [c["entries"][w]["root"] for w in words] == [1, -1, 0, -2]
    assert set(c["pattern_classes"]) == {"<none>", "basic_past", "gerund"}


def test_exact_memory_bypass_with_nonzero_layernorm_bias():
    mem = GroundedMorphMemory(8, 4, 3, dropout=0).eval()
    with torch.no_grad(): mem.norm.bias.fill_(3)
    h = torch.randn(1, 3, 8); mem.override = {"off": True}
    out, _ = mem(h, word_ids=torch.arange(3).unsqueeze(0))
    assert torch.equal(out, h)
