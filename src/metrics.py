"""Criterion-level classification metrics and evidence-sentence retrieval metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import LABELS


def _cls(y_true, y_pred):
    if not y_true:
        return {}
    labels = [l for l in dict.fromkeys(y_true)]                    # labels with gold support
    cm_labels = labels + (["invalid"] if "invalid" in y_pred else [])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "invalid_outputs": int(sum(1 for y in y_pred if y == "invalid")),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": {"labels": cm_labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=cm_labels).tolist()},
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evidence_metrics(gold, pred):
    """gold/pred: lists of id-lists.
    micro  : pooled TP/FP/FN over all records.
    macro  : mean of per-record P/R/F1; if gold AND pred are empty the record scores 1/1/1 (correct
             abstention); if exactly one is empty it scores 0/0/0.
    macro_nonempty_gold : same macro, but only over records whose gold evidence is non-empty."""
    tp = fp = fn = 0
    per, per_ne = [], []
    for g, p in zip(gold, pred):
        g, p = set(g), set(p)
        a, b, c = len(g & p), len(p - g), len(g - p)
        tp, fp, fn = tp + a, fp + b, fn + c
        s = (1.0, 1.0, 1.0) if not g and not p else _prf(a, b, c)
        per.append(s)
        if g:
            per_ne.append(s)
    mp, mr, mf = _prf(tp, fp, fn)
    return {"micro": {"precision": mp, "recall": mr, "f1": mf},
            "macro": dict(zip(("precision", "recall", "f1"), np.mean(per, axis=0).tolist())) if per else {},
            "macro_nonempty_gold": dict(zip(("precision", "recall", "f1"), np.mean(per_ne, axis=0).tolist())) if per_ne else {},
            "n_records": len(per), "n_empty_gold": sum(1 for g in gold if not g),
            "n_empty_pred": sum(1 for p in pred if not p)}


def evaluate_predictions(recs, preds):
    out = {}
    for name, sel in (("overall", None), ("inclusion", "inclusion"), ("exclusion", "exclusion")):
        idx = [i for i, r in enumerate(recs) if sel is None or r["criterion_type"] == sel]
        yt = [recs[i]["label"] for i in idx]
        yp = [preds[i]["label"] for i in idx]
        out[name] = {"classification": _cls(yt, yp),
                     "evidence": evidence_metrics([recs[i]["evidence"] for i in idx],
                                                  [preds[i]["evidence"] for i in idx])}
    return out
