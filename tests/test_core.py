import json
import numpy as np
import pytest
import torch

from src.attention import reference_attention, sdpa_attention, tiled_attention
from src.data import LABELS, build_prompt, parse_evidence, parse_output, parse_sentences, split_patient_ids, target_text
from src.metrics import evaluate_predictions, evidence_metrics


# ---------------- attention
@pytest.mark.parametrize("B,H,Tq,Tk,D,bq,bk", [(1, 2, 16, 16, 8, 4, 4), (2, 3, 37, 37, 16, 8, 16),
                                               (1, 2, 5, 40, 8, 4, 8), (1, 1, 1, 33, 8, 4, 8), (1, 2, 70, 70, 64, 64, 64)])
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_matches_reference(B, H, Tq, Tk, D, bq, bk, causal):
    if causal and Tq > Tk:
        pytest.skip()
    torch.manual_seed(0)
    q, k, v = torch.randn(B, H, Tq, D), torch.randn(B, H, Tk, D), torch.randn(B, H, Tk, D)
    ref = reference_attention(q, k, v, causal)
    til = tiled_attention(q, k, v, causal, block_q=bq, block_k=bk)
    assert torch.allclose(ref, til, atol=1e-5, rtol=1e-4), (ref - til).abs().max()
    assert torch.allclose(ref, sdpa_attention(q, k, v, causal), atol=1e-5, rtol=1e-4)


def test_large_logits_stable():
    q, k, v = torch.randn(1, 1, 8, 4) * 100, torch.randn(1, 1, 8, 4) * 100, torch.randn(1, 1, 8, 4)
    assert torch.isfinite(reference_attention(q, k, v)).all() and torch.isfinite(tiled_attention(q, k, v, block_q=3, block_k=3)).all()


# ---------------- decoding (tiny random Qwen2 models; no downloads)
def _tiny(hidden=64, layers=2, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=300, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.mark.parametrize("n_in,n_out", [(8, 16), (40, 32), (120, 24)])
def test_kv_equals_nocache(n_in, n_out):
    from src.decoding import greedy_kvcache, greedy_nocache
    m = _tiny(); ids = torch.randint(0, 300, (1, n_in))
    a, b = greedy_nocache(m, ids, n_out), greedy_kvcache(m, ids, n_out)
    assert a.tokens == b.tokens and len(a.tokens) == n_out
    assert b.stats["final_cache_len"] == n_in + n_out - 1


def test_kv_matches_hf_generate():
    from src.decoding import greedy_kvcache, hf_generate
    m = _tiny(); ids = torch.randint(0, 300, (1, 30))
    assert greedy_kvcache(m, ids, 20).tokens == hf_generate(m, ids, 20).tokens


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("draft_kind", ["random_draft", "perfect_draft"])
def test_speculative_equals_greedy(k, draft_kind):
    from src.decoding import greedy_kvcache, speculative_greedy
    tgt = _tiny(seed=0)
    drf = _tiny(hidden=32, layers=1, seed=1) if draft_kind == "random_draft" else tgt
    for n_in, n_out in [(6, 20), (50, 33)]:
        ids = torch.randint(0, 300, (1, n_in))
        ref = greedy_kvcache(tgt, ids, n_out)
        sp = speculative_greedy(tgt, drf, ids, n_out, k=k)
        assert sp.tokens == ref.tokens, (draft_kind, k)
        if draft_kind == "perfect_draft":
            assert sp.stats["acceptance_rate"] == 1.0
            assert sp.stats["tokens_per_target_call"] > 1


def test_speculative_eos_stops():
    from src.decoding import greedy_kvcache, speculative_greedy
    tgt = _tiny(); ids = torch.randint(0, 300, (1, 10))
    ref = greedy_kvcache(tgt, ids, 30)
    eos = ref.tokens[7]
    r2 = greedy_kvcache(tgt, ids, 30, eos_id=eos)
    sp = speculative_greedy(tgt, tgt, ids, 30, eos_id=eos, k=4)
    assert sp.tokens == r2.tokens and sp.tokens[-1] == eos


# ---------------- data / metrics
class StubTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        class O: pass
        o = O(); o.input_ids = text.split(); return o


def test_split_is_deterministic_and_disjoint():
    ids = [f"P{i}" for i in range(53)]
    tr, va, te = split_patient_ids(ids)
    assert (len(tr), len(va), len(te)) == (37, 8, 8)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    arr = np.array(sorted(ids)); np.random.default_rng(42).shuffle(arr)
    assert tr == arr[:37].tolist() and te == arr[45:].tolist()
    assert split_patient_ids(ids[::-1]) == (tr, va, te)


def test_parsers():
    s = parse_sentences("0. Aged 5.\n1. Has fever.\n2. No rash.")
    assert s == {0: "Aged 5.", 1: "Has fever.", 2: "No rash."}
    assert parse_evidence("[0, 2]") == [0, 2] and parse_evidence("") == [] and parse_evidence(None) == []
    assert parse_output('{"label":"included","evidence":[1,0]}', "inclusion") == ("included", [0, 1], True)
    assert parse_output('{"label":"included","evidence":[]}', "exclusion")[0] == "invalid"
    assert parse_output("garbage", "inclusion")[0] == "invalid"
    assert json.loads(target_text("excluded", [3, 1])) == {"label": "excluded", "evidence": [1, 3]}


def test_prompt_has_no_gpt4_fields_and_truncates():
    sents = {i: f"sentence number {i} about the patient" for i in range(50)}
    text, dropped = build_prompt(StubTok(), sents, "inclusion", "age > 18", max_prompt_tokens=120)
    assert dropped and dropped == sorted(dropped) and "gpt4" not in text.lower()
    assert "[0]" in text and f"[{dropped[0]}]" not in text


def test_evidence_metrics_empty_handling():
    m = evidence_metrics([[], [1, 2], []], [[], [2, 3], [5]])
    assert m["micro"]["precision"] == pytest.approx(1 / 3) and m["micro"]["recall"] == pytest.approx(0.5)
    assert m["macro"]["f1"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert m["n_empty_gold"] == 2


def test_evaluate_predictions_splits_by_type():
    recs = [{"criterion_type": "inclusion", "label": "included", "evidence": [1]},
            {"criterion_type": "exclusion", "label": "not excluded", "evidence": []},
            {"criterion_type": "exclusion", "label": "excluded", "evidence": [2]}]
    preds = [{"label": "included", "evidence": [1]}, {"label": "not excluded", "evidence": []},
             {"label": "invalid", "evidence": []}]
    m = evaluate_predictions(recs, preds)
    assert m["overall"]["classification"]["accuracy"] == pytest.approx(2 / 3)
    assert m["inclusion"]["classification"]["accuracy"] == 1.0
    assert m["exclusion"]["classification"]["invalid_outputs"] == 1
