"""Dataset loading, patient-grouped split (seed 42, 37/8/8), prompt building, output parsing.

Only the *expert* columns are ever used. Every GPT-4 column (gpt4_*) is ignored by construction:
`make_records` copies a fixed whitelist of fields, so model predictions cannot leak into inputs.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

DATASET_ID = "ncbi/TrialGPT-Criterion-Annotations"
SEED, N_TRAIN, N_VAL = 42, 37, 8
DATA_DIR = Path("data")

LABELS = {
    "inclusion": ["included", "not included", "not enough information", "not applicable"],
    "exclusion": ["excluded", "not excluded", "not enough information", "not applicable"],
}
ALL_LABELS = ["included", "not included", "excluded", "not excluded",
              "not enough information", "not applicable"]
ALIASES = {"not enough info": "not enough information", "insufficient information": "not enough information",
           "insufficient": "not enough information", "n/a": "not applicable", "na": "not applicable"}

# first existing column wins. Run `python -m src.data inspect` to confirm the mapping on your copy.
COLUMN_CANDIDATES = {
    "patient_id": ["patient_id", "patientid", "pid"],
    "trial_id": ["trial_id", "nct_id", "trialid"],
    "patient": ["patient", "patient_note", "patient_description", "note"],
    "criterion_type": ["criterion_type", "type", "criteria_type"],
    "criterion_text": ["criterion_text", "criterion", "criteria_text"],
    "label": ["expert_eligibility", "expert_label", "expert_eligibility_label", "eligibility"],
    "evidence": ["expert_sentences", "expert_evidence", "expert_sentence_ids", "evidence_sentences"],
}

# ----------------------------------------------------------------------------- loading
def load_raw(local=None):
    import pandas as pd
    if local:
        p = str(local)
        if p.endswith(".csv"):
            return pd.read_csv(p)
        return pd.read_json(p, lines=p.endswith(".jsonl"))
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID)
    return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)


def resolve_columns(df):
    lower = {c.lower(): c for c in df.columns}
    cols = {}
    for key, cands in COLUMN_CANDIDATES.items():
        hit = next((lower[c] for c in cands if c in lower), None)
        if hit is None and key not in ("trial_id",):
            raise KeyError(f"No column for '{key}'. Available: {list(df.columns)}. "
                           f"Edit COLUMN_CANDIDATES in src/data.py.")
        cols[key] = hit
    return cols


def norm_label(x):
    s = re.sub(r"\s+", " ", str(x).strip().lower())
    return ALIASES.get(s, s)


def parse_evidence(v):
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return sorted({int(x) for x in v})
    return sorted({int(x) for x in re.findall(r"\d+", str(v))})


def parse_sentences(note):
    """Return {sentence_id: text}. Honors the dataset's '0. sentence' numbering, else splits."""
    lines = [l for l in str(note).split("\n") if l.strip()]
    ms = [re.match(r"^\s*(\d+)\s*[.:)\]]\s*(.*)$", l) for l in lines]
    if lines and sum(m is not None for m in ms) >= 0.8 * len(lines):
        return {int(m.group(1)): m.group(2).strip() for m in ms if m and m.group(2).strip()}
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", str(note).strip().replace("\n", " "))
    return {i: p.strip() for i, p in enumerate(p for p in parts if p.strip())}


def make_records(df, cols):
    recs, dropped = [], 0
    for _, r in df.iterrows():
        ctype = "inclusion" if str(r[cols["criterion_type"]]).lower().startswith("incl") else "exclusion"
        label = norm_label(r[cols["label"]])
        if label not in LABELS[ctype]:
            dropped += 1
            continue
        recs.append({
            "patient_id": str(r[cols["patient_id"]]),
            "trial_id": str(r[cols["trial_id"]]) if cols["trial_id"] else "",
            "criterion_type": ctype,
            "criterion_text": str(r[cols["criterion_text"]]).strip(),
            "label": label,
            "evidence": parse_evidence(r[cols["evidence"]]),
            "sentences": parse_sentences(r[cols["patient"]]),
        })
    if dropped:
        print(f"[warn] dropped {dropped} rows with labels outside the allowed set")
    return recs, df[cols["patient_id"]]


# ----------------------------------------------------------------------------- split
def split_patient_ids(pids):
    """Sort unique IDs, shuffle with default_rng(42), first 37 train / next 8 val / rest test."""
    ids = np.array(sorted(set(pids)))
    np.random.default_rng(SEED).shuffle(ids)
    return ids[:N_TRAIN].tolist(), ids[N_TRAIN:N_TRAIN + N_VAL].tolist(), ids[N_TRAIN + N_VAL:].tolist()


def prepare(local=None, out=DATA_DIR):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = load_raw(local)
    cols = resolve_columns(df)
    recs, native_pids = make_records(df, cols)
    tr, va, te = split_patient_ids(native_pids.tolist())
    sets = {"train": {str(x) for x in tr}, "val": {str(x) for x in va}, "test": {str(x) for x in te}}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "patient leakage"
    manifest = {"seed": SEED, "dataset": DATASET_ID, "columns_used": cols, "splits": {}}
    for name, pset in sets.items():
        part = [r for r in recs if r["patient_id"] in pset]
        with open(out / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        manifest["splits"][name] = {"n_patients": len(pset), "n_records": len(part), "patient_ids": sorted(pset)}
        print(f"{name:5s}: {len(pset):3d} patients, {len(part):5d} criterion records")
    assert sum(len(s) for s in sets.values()) == len(set(map(str, native_pids))), "some patients unassigned"
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest ->", out / "split_manifest.json")
    if (len(tr), len(va), len(te)) != (37, 8, 8):
        print(f"[warn] expected 37/8/8 patients, got {len(tr)}/{len(va)}/{len(te)}")


def load_split(name, data_dir=DATA_DIR):
    recs = []
    for line in open(Path(data_dir) / f"{name}.jsonl"):
        r = json.loads(line)
        r["sentences"] = {int(k): v for k, v in r["sentences"].items()}
        recs.append(r)
    return recs


# ----------------------------------------------------------------------------- prompts
SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility annotator. You receive a patient note whose sentences are "
    "numbered like [3], one trial criterion and its type. Decide how the patient relates to the criterion "
    "using ONLY the note, and cite the sentence IDs that support the decision. If the note does not settle "
    "the criterion answer 'not enough information' - absence of evidence is not proof either way. "
    'Reply with JSON only: {"label": <label>, "evidence": [<sentence ids>]}.'
)


def format_patient(sentences):
    return "\n".join(f"[{i}] {t}" for i, t in sorted(sentences.items()))


def user_message(sentences, ctype, ctext):
    labels = ", ".join(f'"{l}"' for l in LABELS[ctype])
    return (f"Criterion type: {ctype}\nAllowed labels: {labels}\n\n"
            f"Patient note:\n{format_patient(sentences)}\n\nCriterion: {ctext}")


def build_prompt(tokenizer, sentences, ctype, ctext, max_prompt_tokens=None):
    """Chat-formatted prompt. If too long, drop trailing patient sentences; return the dropped IDs."""
    keep = sorted(sentences)
    dropped = []

    def render(ids):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message({i: sentences[i] for i in ids}, ctype, ctext)}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    text = render(keep)
    if max_prompt_tokens:
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        while n > max_prompt_tokens and len(keep) > 1:
            dropped.insert(0, keep.pop())
            text = render(keep)
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
    return text, dropped


def target_text(label, evidence):
    return json.dumps({"label": label, "evidence": sorted(evidence)}, separators=(",", ":"))


def parse_output(text, ctype):
    """-> (label | 'invalid', evidence_ids, valid)"""
    label, ev, valid = "invalid", [], False
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            lab = norm_label(obj.get("label", ""))
            if lab in LABELS[ctype]:
                label, valid = lab, True
            ev = [int(x) for x in obj.get("evidence", []) if str(x).strip().isdigit()]
        except Exception:
            pass
    return label, sorted(set(ev)), valid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "inspect"])
    ap.add_argument("--local", default=None, help="optional local csv/json/jsonl copy of the dataset")
    a = ap.parse_args()
    if a.cmd == "inspect":
        df = load_raw(a.local)
        print("columns:", list(df.columns)); print("rows:", len(df))
        print(df.iloc[0].to_dict())
        print("column mapping ->", resolve_columns(df))
    else:
        prepare(a.local)
