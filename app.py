"""Task 6: browser application (Gradio).  Start:  python app.py   ->  http://127.0.0.1:7860
Env: BASE_MODEL, ADAPTER_DIR, DRAFT_MODEL, PORT, SHARE=1"""
import glob
import html
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from src import attention as A
from src import decoding as D
from src import pipeline as P
from src.bench import measure
from src.data import build_prompt, parse_sentences

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.environ.get("ADAPTER_DIR", "outputs/lora/best")
DRAFT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DISCLAIMER = ("**Educational screening prototype - NOT a clinical enrollment decision.** Outputs come from a small "
              "fine-tuned language model, can be wrong, and must be reviewed by qualified clinicians.")
EXAMPLE_NOTE = ("A 58-year-old man presents with type 2 diabetes for 10 years. His HbA1c last month was 8.4%. "
                "He takes metformin and has never smoked. He denies any history of heart failure. "
                "Creatinine is 1.1 mg/dL. He has not undergone any surgery in the last year.")
EXAMPLE_CRIT = """Inclusion Criteria:
 * Age between 18 and 75 years
 * Diagnosed with type 2 diabetes mellitus
 * HbA1c above 7.5%
Exclusion Criteria:
 * History of heart failure
 * Current insulin therapy
 * Estimated GFR below 30 mL/min"""
S = {}


def models(need_draft=False):
    if "m" not in S:
        ad = ADAPTER if Path(ADAPTER).exists() else None
        if ad is None:
            gr.Warning(f"Adapter '{ADAPTER}' not found - running the BASE model (no fine-tuning).")
        S["m"], S["tok"] = P.load_model(BASE, ad)
        S["adapter"] = ad
    if need_draft and "d" not in S:
        S["d"] = P.load_model(DRAFT)[0]
    return S["m"], S["tok"], S.get("d")


def validate(note, crit_text):
    if not note or len(note.strip()) < 20:
        return None, "Patient note is empty or too short (min 20 characters)."
    if len(note) > 20000:
        return None, "Patient note is too long (max 20,000 characters)."
    crit = P.parse_trial_criteria(crit_text or "")
    if not crit:
        return None, ("No criteria found. Use 'Inclusion Criteria:' / 'Exclusion Criteria:' headings, "
                      "each criterion on a line starting with *, - or a number.")
    if len(crit) > 40:
        return None, f"Too many criteria ({len(crit)}); max is 40 per run."
    return crit, None


def render_note(sents, cited):
    return "\n".join(f"{'**' if i in cited else ''}[{i}] {t}{'**' if i in cited else ''}  " for i, t in sorted(sents.items()))


def make_report(note, sents, rows, overall, mode, k, t):
    e = html.escape
    tr = "".join(
        f"<tr><td>{e(r['criterion_type'])}</td><td>{e(r['criterion'])}</td><td><b>{e(r['prediction'])}</b></td>"
        f"<td>{', '.join(map(str, r['evidence_ids'])) or '-'}</td>"
        f"<td>{'<br>'.join(e(x) for x in r['evidence_text']) or '-'}</td><td>{e(r['explanation'])}</td></tr>" for r in rows)
    doc = f"""<html><head><meta charset='utf-8'><title>Eligibility screening report</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse}}td,th{{border:1px solid #888;padding:6px;vertical-align:top}}
.warn{{background:#fff3cd;padding:10px;border:1px solid #e0c060}}</style></head><body>
<h1>Eligibility screening report</h1><p class='warn'>{e(DISCLAIMER.replace('**', ''))}</p>
<h2>Provenance</h2><ul><li>Base model: {e(BASE)}</li><li>Adapter: {e(str(S.get('adapter')))}</li><li>Decoding: {e(mode)} (greedy{', k=' + str(k) if mode == 'speculative' else ''})</li>
<li>Generated: {e(t)}</li></ul><h2>Overall</h2><p>{e(overall)} <i>(rule-based aggregation of criterion-level predictions, not a labelled outcome)</i></p>
<h2>Criterion-level results</h2><table><tr><th>Type</th><th>Criterion</th><th>Prediction</th><th>Sentence IDs</th><th>Evidence</th><th>Explanation</th></tr>{tr}</table>
<h2>Patient note (numbered)</h2><pre>{e(chr(10).join(f'[{i}] {x}' for i, x in sorted(sents.items())))}</pre></body></html>"""
    p = Path(tempfile.mkdtemp()) / f"screening_report_{int(time.time())}.html"
    p.write_text(doc, encoding="utf-8")
    return str(p)


def run_screen(note, crit_text, mode, max_new, k):
    crit, err = validate(note, crit_text)
    if err:
        return pd.DataFrame(), f"⚠️ {err}", "", None
    try:
        m, tok, d = models(mode in ("speculative", "hf_assisted"))
        sents, rows = P.screen(m, tok, note, crit, mode, d, int(max_new), int(k))
    except Exception as ex:  # keep the UI alive
        return pd.DataFrame(), f"⚠️ Inference failed: {ex}", "", None
    overall = P.aggregate(rows)
    df = pd.DataFrame([{"type": r["criterion_type"], "criterion": r["criterion"], "prediction": r["prediction"],
                        "sentence_ids": ", ".join(map(str, r["evidence_ids"])), "evidence": " | ".join(r["evidence_text"]),
                        "explanation": r["explanation"], "latency_s": r["latency_s"],
                        "truncated": bool(r["dropped_by_truncation"])} for r in rows])
    cited = {i for r in rows for i in r["evidence_ids"]}
    path = make_report(note, sents, rows, overall, mode, k, time.strftime("%Y-%m-%d %H:%M:%S"))
    return df, f"### {overall}\n\n{DISCLAIMER}", render_note(sents, cited), path


def run_compare(note, crit_text, which, modes, max_new, k, repeats):
    crit, err = validate(note, crit_text)
    if err:
        return pd.DataFrame(), f"⚠️ {err}"
    if not modes:
        return pd.DataFrame(), "⚠️ Select at least one decoding mode."
    try:
        need_d = any(x in modes for x in ("speculative", "hf_assisted"))
        m, tok, d = models(need_d)
        ctype, ctext = crit[min(int(which), len(crit)) - 1]
        prompt, _ = build_prompt(tok, parse_sentences(note), ctype, ctext, 3500)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(next(m.parameters()).device)
        dev = str(ids.device); ref = None; rows = []
        for mode in [x for x in D.MODES if x in modes]:
            res, st = measure(lambda: D.generate(mode, m, ids, int(max_new), tok.eos_token_id, d, int(k)), dev, int(repeats))
            ref = ref or res.tokens
            rows.append({"mode": mode, "prompt_tokens": ids.size(1), "new_tokens": len(res.tokens),
                         "prefill_s": round(st["prefill_s"], 4), "decode_s": round(st["decode_s"], 4),
                         "tokens_per_s": round(st["tok_per_s"], 2), "peak_mem_mb": round(st["peak_mem_mb"], 1),
                         "acceptance_rate": round(res.stats.get("acceptance_rate", float("nan")), 3) if res.stats else None,
                         "same_tokens_as_first_mode": res.tokens == ref})
        return pd.DataFrame(rows), f"Criterion tested: *{ctype}* - {ctext}  \n(first selected mode is the token-equality reference)"
    except Exception as ex:
        return pd.DataFrame(), f"⚠️ Benchmark failed: {ex}"


def run_attn(T, D_, H, block, repeats):
    try:
        rows = A.benchmark(shapes=[(1, int(H), int(T), int(D_))], repeats=int(repeats), block=int(block))
        return pd.DataFrame(rows)
    except Exception as ex:
        return pd.DataFrame([{"error": str(ex)}])


def load_results():
    out = []
    for f in sorted(glob.glob("results/eval_*.json")):
        j = json.load(open(f)); m = j["metrics"]
        for part in ("overall", "inclusion", "exclusion"):
            c, e = m[part]["classification"], m[part]["evidence"]
            out.append({"run": Path(f).stem, "subset": part, "n": c.get("n"), "accuracy": round(c.get("accuracy", 0), 3),
                        "macro_f1": round(c.get("macro_f1", 0), 3), "evid_micro_F1": round(e["micro"]["f1"], 3),
                        "evid_macro_F1": round(e["macro"].get("f1", 0), 3)})
    ev = pd.DataFrame(out)
    bl = []
    for f in sorted(glob.glob("results/bench_*.json")):
        for r in json.load(open(f))["rows"]:
            bl.append({"file": Path(f).stem, **{k: v for k, v in r.items() if k != "spec_stats"}})
    return ev, pd.DataFrame(bl)


with gr.Blocks(title="TrialMatch - eligibility screening prototype") as demo:
    gr.Markdown("# TrialMatch: criterion-level clinical-trial eligibility screening\n" + DISCLAIMER)
    with gr.Tab("Screening"):
        with gr.Row():
            note = gr.Textbox(EXAMPLE_NOTE, label="Patient note", lines=10)
            crit = gr.Textbox(EXAMPLE_CRIT, label="Trial eligibility criteria", lines=10)
        with gr.Row():
            mode = gr.Dropdown(D.MODES, value="kv_cache", label="Decoding mode")
            max_new = gr.Slider(16, 128, 64, step=8, label="max new tokens")
            kk = gr.Slider(1, 8, 4, step=1, label="draft length k (speculative)")
        go = gr.Button("Run screening", variant="primary")
        overall = gr.Markdown()
        table = gr.Dataframe(label="Criterion-level results", wrap=True)
        gr.Markdown("**Patient note with cited sentences in bold**")
        shown = gr.Markdown()
        rep = gr.File(label="Download screening report (HTML)")
        go.click(run_screen, [note, crit, mode, max_new, kk], [table, overall, shown, rep])
    with gr.Tab("Inference comparison"):
        gr.Markdown("Runs the **same prompt** (first-criterion by default) through each selected decoding path, live.")
        which = gr.Number(1, label="Criterion number (from the Screening tab list)", precision=0)
        modes = gr.CheckboxGroup(D.MODES, value=["no_cache", "kv_cache"], label="Modes (first = reference)")
        with gr.Row():
            mn = gr.Slider(8, 128, 48, step=8, label="max new tokens")
            k2 = gr.Slider(1, 8, 4, step=1, label="k")
            rp = gr.Slider(1, 5, 2, step=1, label="repeats")
        b = gr.Button("Run comparison", variant="primary")
        info = gr.Markdown(); cmp_ = gr.Dataframe()
        b.click(run_compare, [note, crit, which, modes, mn, k2, rp], [cmp_, info])
    with gr.Tab("Attention lab"):
        gr.Markdown("Reference vs tiled online-softmax vs PyTorch SDPA (errors measured against a float64 oracle).")
        with gr.Row():
            T = gr.Slider(32, 2048, 512, step=32, label="sequence length"); Dm = gr.Slider(16, 128, 64, step=16, label="head dim")
            H = gr.Slider(1, 16, 8, step=1, label="heads"); Bk = gr.Slider(16, 256, 64, step=16, label="tile size")
            R = gr.Slider(1, 10, 3, step=1, label="repeats")
        ab = gr.Button("Run attention benchmark"); at = gr.Dataframe()
        ab.click(run_attn, [T, Dm, H, Bk, R], at)
    with gr.Tab("Saved experiment results"):
        rb = gr.Button("Load results/*.json"); t1 = gr.Dataframe(label="Evaluation"); t2 = gr.Dataframe(label="Benchmarks")
        rb.click(load_results, None, [t1, t2])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), share=bool(os.environ.get("SHARE")))
