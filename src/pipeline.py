"""Model loading, trial-criteria parsing, criterion-level screening, rule-based aggregation."""
import re
import torch

from .data import build_prompt, parse_output, parse_sentences
from .decoding import generate


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device):
    return torch.float16 if device == "cuda" else torch.float32


def load_model(base, adapter=None, device=None, dtype=None, merge=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or default_device()
    dtype = dtype or default_dtype(device)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    return model.eval(), tok


def parse_trial_criteria(text):
    """Parse ClinicalTrials.gov style text ('Inclusion Criteria:' / 'Exclusion Criteria:' + bullets)."""
    out, ctype, buf = [], None, None

    def flush():
        nonlocal buf
        if buf and ctype:
            out.append((ctype, re.sub(r"\s+", " ", buf).strip()))
        buf = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().rstrip(":").strip()
        if low.startswith("inclusion criteria"):
            flush(); ctype = "inclusion"; continue
        if low.startswith("exclusion criteria"):
            flush(); ctype = "exclusion"; continue
        if ctype is None:
            continue
        m = re.match(r"^(\*|-|•|\d+[.)])\s+(.*)", line)
        if m:
            flush(); buf = m.group(2)
        else:
            buf = f"{buf} {line}" if buf else line
    flush()
    return out


def explain(label, ctype, ev_ids, sentences):
    """Template explanation built ONLY from the predicted label and the cited note sentences."""
    cited = "; ".join(f'[{i}] "{sentences[i]}"' for i in ev_ids)
    if label == "invalid":
        return "The model output could not be parsed; treat this criterion as unresolved."
    if label == "not enough information":
        return ("The note does not settle this criterion; missing information is not treated as "
                "eligibility or ineligibility." + (f" Related sentences: {cited}" if cited else ""))
    if label == "not applicable":
        return "The criterion was judged not applicable to this patient." + (f" Basis: {cited}" if cited else "")
    if ev_ids:
        return f"Predicted '{label}' for this {ctype} criterion based on: {cited}"
    return f"Predicted '{label}' but the model cited no sentence - verify manually."


def screen(model, tok, patient_note, criteria, mode="kv_cache", draft=None, max_new_tokens=64, k=4,
           max_prompt_tokens=3500):
    sentences = parse_sentences(patient_note)
    dev = next(model.parameters()).device
    rows = []
    for ctype, ctext in criteria:
        prompt, dropped = build_prompt(tok, sentences, ctype, ctext, max_prompt_tokens)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        res = generate(mode, model, ids, max_new_tokens, tok.eos_token_id, draft, k)
        raw = tok.decode(res.tokens, skip_special_tokens=True)
        label, ev, valid = parse_output(raw, ctype)
        bad = [e for e in ev if e not in sentences]
        ev = [e for e in ev if e in sentences]
        rows.append({
            "criterion_type": ctype, "criterion": ctext, "prediction": label, "valid_output": valid,
            "evidence_ids": ev, "evidence_text": [sentences[i] for i in ev],
            "explanation": explain(label, ctype, ev, sentences),
            "dropped_by_truncation": dropped, "ignored_invalid_ids": bad, "raw_output": raw,
            "prompt_tokens": ids.size(1), "new_tokens": len(res.tokens), "latency_s": round(res.total_s, 3),
            "spec_stats": res.stats if mode == "speculative" else {},
        })
    return sentences, rows


def aggregate(rows):
    """Explicit rule-based preliminary outcome (NOT a ground-truth label)."""
    if any(r["prediction"] == "excluded" for r in rows):
        return "Preliminary: likely NOT eligible (at least one exclusion criterion predicted 'excluded')."
    if any(r["prediction"] == "not included" for r in rows):
        return "Preliminary: likely NOT eligible (at least one inclusion criterion predicted 'not included')."
    unresolved = [r for r in rows if r["prediction"] in ("not enough information", "invalid")]
    if unresolved:
        return f"Preliminary: NEEDS MANUAL REVIEW ({len(unresolved)} criteria lack enough information)."
    return "Preliminary: no blocking criterion found (all applicable inclusion met, no exclusion triggered)."
