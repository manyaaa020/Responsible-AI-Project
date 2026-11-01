"""
Extensions that deepen the fairness analysis.

1. bootstrap_group_fpr      95% confidence intervals for each group's false-positive rate,
                            and for the baseline-to-final improvement (paired bootstrap).
                            Small groups have only ~20 harmless test posts, so this says which
                            gaps are real and which could be noise.
2. inprocessing_models      In-processing mitigation with Fairlearn's ExponentiatedGradient
                            (Agarwal et al., 2018): the classifier is trained under a fairness
                            constraint (false-positive-rate parity or equalized odds), completing
                            pre- / in- / post-processing coverage.
3. metricframe_crosscheck   Recomputes per-group FPR with Fairlearn's MetricFrame to confirm our
                            hand-written metrics agree with a standard library.
4. annotator_disagreement   HateXplain has three annotators per post. Measures whether the model's
                            errors, and the over-flagged Jewish posts, concentrate on posts the
                            annotators themselves disagreed about (bias in the labels).
5. model_card               A one-page model card (Mitchell et al., 2019) generated from the results.
"""
import json
import os
from collections import Counter

import numpy as np
from fairlearn.metrics import MetricFrame, false_positive_rate
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient, FalsePositiveRateParity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from .data import DATA_DIR, GROUPS
from .fairness import bias_report


# ---------------------------------------------------------------- 1 bootstrap CIs
def _group_fpr(y, pred, mask):
    m = mask & (y == 0)
    return pred[m].mean() if m.any() else np.nan


def bootstrap_group_fpr(te, scores_by_model, B=1000, seed=0):
    """scores_by_model: {"baseline": scores, "final": scores}, thresholds already mapped to 0.5.
    Returns per model, per group: point estimate and 95% CI of FPR; plus the paired CI of
    (baseline - final) for each group."""
    rng = np.random.default_rng(seed)
    y = te["toxic"].values
    n = len(y)
    masks = {g: te[f"g_{g}"].values for g in GROUPS}
    masks["background"] = ~te["any_identity"].values
    preds = {k: (np.asarray(s) >= 0.5).astype(int) for k, s in scores_by_model.items()}
    boot = {k: {g: [] for g in masks} for k in preds}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for k, p in preds.items():
            pb = p[idx]
            for g, m in masks.items():
                boot[k][g].append(_group_fpr(yb, pb, m[idx]))
    out = {}
    for k, p in preds.items():
        out[k] = {}
        for g, m in masks.items():
            arr = np.array(boot[k][g], dtype=float)
            out[k][g] = {"fpr": float(_group_fpr(y, p, m)),
                         "lo": float(np.nanpercentile(arr, 2.5)),
                         "hi": float(np.nanpercentile(arr, 97.5)),
                         "n_harmless": int((m & (y == 0)).sum())}
    if "baseline" in preds and "final" in preds:
        out["improvement"] = {}
        for g in masks:
            d = np.array(boot["baseline"][g], dtype=float) - np.array(boot["final"][g], dtype=float)
            lo, hi = np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
            out["improvement"][g] = {
                "diff": out["baseline"][g]["fpr"] - out["final"][g]["fpr"],
                "lo": float(lo), "hi": float(hi),
                "significant": bool(lo > 0 or hi < 0)}
    return out


# ---------------------------------------------------------------- 2 in-processing
def sensitive_feature(df):
    """One label per post for Fairlearn: the single group mentioned, 'multiple', or 'none'."""
    return df["groups"].apply(lambda gs: gs[0] if len(gs) == 1 else ("multiple" if gs else "none")).values


_MATRICES = {}


class IndexedLR(BaseEstimator, ClassifierMixin):
    """Logistic regression that receives row indices instead of features.

    Fairlearn's reductions only accept dense arrays, and our 90k-column sparse TF-IDF matrix would
    be ~11 GB dense. Fairlearn therefore passes a column of row numbers, and this estimator looks
    the rows up in the sparse matrix, so the constrained model uses exactly the same features."""

    def __init__(self, key="train", C=4.0):
        self.key, self.C = key, C

    def fit(self, X, y, sample_weight=None):
        rows = np.asarray(X)[:, 0].astype(int)
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(
            _MATRICES[self.key][rows], y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X):
        rows = np.asarray(X)[:, 0].astype(int)
        return self.lr_.predict(_MATRICES[self.key][rows])


class ReductionModel:
    """A fitted ExponentiatedGradient exposed like ModerationModel (predict_proba on texts).
    The score is the expected probability under Fairlearn's randomized classifier: each
    predictor's probability weighted by the weight the reduction gave it."""

    def __init__(self, featurizer, eg, name):
        self.featurizer, self.eg, self.name = featurizer, eg, name

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = self.featurizer._features(list(texts))
        w = self.eg.weights_
        return sum(w[t] * self.eg.predictors_[t].lr_.predict_proba(X)[:, 1] for t in w.index if w[t] > 0)


def inprocessing_models(featurizer, tr, eps=0.02, max_iter=50):
    """featurizer: a ModerationModel whose vectorizers are already fitted on the training text."""
    _MATRICES["train"] = featurizer._features(list(tr["text"]))
    rows = np.arange(len(tr)).reshape(-1, 1)
    y = tr["toxic"].values
    sf = sensitive_feature(tr)
    models = {}
    for name, constraint in [("FPR parity", FalsePositiveRateParity(difference_bound=eps)),
                             ("Equalized odds", EqualizedOdds(difference_bound=eps))]:
        eg = ExponentiatedGradient(IndexedLR(), constraints=constraint, eps=eps, max_iter=max_iter)
        eg.fit(rows, y, sensitive_features=sf)
        models[name] = ReductionModel(featurizer, eg, name)
    return models


# ---------------------------------------------------------------- 3 MetricFrame cross-check
def metricframe_crosscheck(te, scores):
    """Per-group FPR from Fairlearn's MetricFrame vs our bias_report, at the 0.5 boundary."""
    y = te["toxic"].values
    pred = (np.asarray(scores) >= 0.5).astype(int)
    ours = bias_report(te, scores)["groups"]
    rows, max_diff = [], 0.0
    for g in ours:
        mf = MetricFrame(metrics=false_positive_rate, y_true=y, y_pred=pred,
                         sensitive_features=te[f"g_{g}"].values)
        theirs = float(mf.by_group.loc[True])
        diff = abs(theirs - ours[g]["fpr"])
        max_diff = max(max_diff, diff)
        rows.append({"group": g, "ours": ours[g]["fpr"], "fairlearn": theirs, "abs_diff": diff})
    return {"rows": rows, "max_abs_diff": max_diff}


# ---------------------------------------------------------------- 4 annotator disagreement
def toxic_votes(df):
    """Number of the three annotators who called each post hate speech or offensive."""
    raw = json.load(open(os.path.join(DATA_DIR, "dataset.json")))
    votes = {pid: sum(a["label"] in ("hatespeech", "offensive") for a in post["annotators"])
             for pid, post in raw.items()}
    return df["post_id"].map(votes).values


def annotator_disagreement(te, scores):
    y = te["toxic"].values
    pred = (np.asarray(scores) >= 0.5).astype(int)
    v = toxic_votes(te)
    split = (v == 1) | (v == 2)          # annotators disagreed on toxic vs not
    err = pred != y
    harmless = y == 0
    fp = harmless & (pred == 1)
    ident = te["any_identity"].values
    jew = te["g_jewish"].values
    out = {
        "share_split_all": float(split.mean()),
        "error_rate_unanimous": float(err[~split].mean()),
        "error_rate_split": float(err[split].mean()),
        "fpr_unanimous_harmless": float(pred[harmless & ~split].mean()),
        "fpr_split_harmless": float(pred[harmless & split].mean()),
        "split_share_harmless": float(split[harmless].mean()),
        "split_share_false_positives": float(split[fp].mean()),
        "split_share_identity_false_positives": float(split[fp & ident].mean()) if (fp & ident).any() else None,
        "jewish_harmless_n": int((harmless & jew).sum()),
        "jewish_harmless_split_share": float(split[harmless & jew].mean()),
        "background_harmless_split_share": float(split[harmless & ~ident].mean()),
        "vote_distribution_test": {int(k): int(c) for k, c in sorted(Counter(v).items())},
    }
    return out


# ---------------------------------------------------------------- 5 model card
def model_card(metrics, ext, path):
    f = metrics["final"]
    o = f["bias_report"]["overall"]
    groups = f["bias_report"]["groups"]
    ev = f["evasion"]
    bands = f["bands"]
    ci = ext["bootstrap"]["final"]
    lines = [
        "# Model card: responsible toxicity moderator",
        "",
        "## Model details",
        "- TF-IDF (word 1-2 grams, character 2-5 grams) with logistic regression, trained with text "
        "normalisation and adversarial augmentation of both classes.",
        f"- Operating threshold calibrated on validation data ({f['global_threshold']:.3f}); one-sided group "
        f"thresholds: " + ", ".join(f"{g} {t:.2f}" for g, t in f["group_thresholds"].items() if t > 0.5) + ".",
        "- Three-band routing: allow below 0.40, human review 0.40-0.85, remove at 0.85 and above.",
        "",
        "## Intended use",
        "- Screening English social media posts for hate speech and offensive content, with a person "
        "reviewing every uncertain case and an appeal route for every removal.",
        "- Not intended for: fully automatic removal, non-English or code-mixed text, images or memes, "
        "or decisions about people outside content moderation.",
        "",
        "## Training and evaluation data",
        "- HateXplain (Mathew et al., 2021): 19,229 Twitter and Gab posts with majority labels; official "
        "split 15,383 / 1,922 / 1,924. Collected by keyword search, so identity terms are over-represented "
        "in toxic posts.",
        "",
        "## Performance (test split)",
        f"- AUC {o['auc']:.3f}, F1 {o['f1']:.3f}, final bias score {o['final_bias_score']:.3f}, "
        f"mean BPSN AUC {o['pm_bpsn_auc']:.3f}.",
        "",
        "| Group | Harmless test posts | False-positive rate | 95% CI | False-negative rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g, v in groups.items():
        c = ci[g]
        lines.append(f"| {g} | {c['n_harmless']} | {v['fpr']:.0%} | {c['lo']:.0%}-{c['hi']:.0%} | {v['fnr']:.0%} |")
    b = ci["background"]
    lines += [f"| no group named | {b['n_harmless']} | {b['fpr']:.0%} | {b['lo']:.0%}-{b['hi']:.0%} | - |", "",
              "## Robustness",
              f"- Attack success rates: leetspeak {ev['leetspeak']:.0%}, look-alike letters {ev['homoglyph']:.0%}, "
              f"invisible characters {ev['zero_width']:.0%}, spaced letters {ev['spacing']:.0%}, padding "
              f"{ev['benign_pad']:.0%}, adaptive (2 words) {ev['greedy_2']:.0%}, adaptive (30%) {ev['greedy']:.0%}.",
              "", "## Routing on the test set",
              f"- Allow {bands['allow']['share']:.0%}, human review {bands['human review']['share']:.0%} "
              f"({bands['human review']['toxic_rate']:.0%} toxic), remove {bands['remove']['share']:.0%} "
              f"({bands['remove']['toxic_rate']:.0%} toxic).",
              "", "## Ethical considerations and limitations",
              "- Raising the Jewish-group threshold lowers wrong flags but raises missed hate against that "
              "group; the human review band is what keeps this acceptable.",
              "- Adaptive attackers still evade the model most of the time; rate limits and human review are required.",
              "- Group tags come from a lexicon of neutral terms and miss coded or implicit references.",
              "- Several groups have fewer than 60 harmless test posts, so their rates carry wide intervals (table above).",
              "- Retraining on user reports needs the label-drift check and trigger scan described in the report."]
    open(path, "w").write("\n".join(lines) + "\n")
