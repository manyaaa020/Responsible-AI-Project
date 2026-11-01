"""
Fairness measurement and mitigation.

Metrics
  Subgroup AUC   AUC using only posts that mention the group. Low = the model can't
                 separate toxic from non-toxic posts about this group.
  BPSN AUC       Background-Positive, Subgroup-Negative. Low = non-toxic posts that
                 mention the group score higher than toxic posts that don't:
                 the "flagging 'I am gay'" failure (false positives).
  BNSP AUC       Background-Negative, Subgroup-Positive. Low = toxic posts about the
                 group score lower than clean background posts (missed hate).
  Final bias score (Borkan et al., 2019, used in the Jigsaw competition):
                 0.25 * overall AUC + 0.75 * mean of the three generalised means (p = -5)
  FPR / FNR per group at the operating threshold, and the max gap to the no-identity
                 background (equalized-odds style).
  Template FPR   share of neutral sentences ("i am a muslim person") that get flagged.
  Counterfactual token fairness gap (Garg et al., 2019)
                 mean |score(x) - score(x with identity term swapped)|. 0 = the model's
                 score does not depend on which group is mentioned.

Mitigation
  reweigh              pre-processing: Kamiran & Calders reweighing on (group, label),
                       so identity mentions are no longer statistically tied to toxicity.
  counterfactual_augment pre-processing: copies of identity-mentioning training posts
                       with the term swapped to another group, label unchanged.
  group_thresholds     post-processing: pick a per-group threshold so every group's
                       FPR on validation data is close to the background FPR.
"""
import random

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from .data import GROUPS, IDENTITY_TERMS, TERM_TO_GROUP


def _auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def power_mean(values, p=-5):
    v = np.array([x for x in values if not np.isnan(x)])
    return float(np.power(np.mean(np.power(v, p)), 1 / p))


def bias_report(df, scores, threshold=0.5, min_n=15):
    y = df["toxic"].values
    s = np.asarray(scores)
    pred = (s >= threshold).astype(int)
    bg = ~df["any_identity"].values
    out = {"overall": {
        "auc": _auc(y, s), "f1": f1_score(y, pred), "accuracy": accuracy_score(y, pred),
        "fpr_background": float(pred[bg & (y == 0)].mean()),
        "fnr_background": float(1 - pred[bg & (y == 1)].mean()),
    }, "groups": {}}
    for g in GROUPS:
        m = df[f"g_{g}"].values
        if (m & (y == 0)).sum() < min_n or (m & (y == 1)).sum() < min_n:
            continue
        bpsn = (m & (y == 0)) | (~m & (y == 1))
        bnsp = (m & (y == 1)) | (~m & (y == 0))
        out["groups"][g] = {
            "n": int(m.sum()),
            "subgroup_auc": _auc(y[m], s[m]),
            "bpsn_auc": _auc(y[bpsn], s[bpsn]),
            "bnsp_auc": _auc(y[bnsp], s[bnsp]),
            "fpr": float(pred[m & (y == 0)].mean()),
            "fnr": float(1 - pred[m & (y == 1)].mean()),
        }
    G = out["groups"].values()
    pm = {k: power_mean([g[k] for g in G]) for k in ("subgroup_auc", "bpsn_auc", "bnsp_auc")}
    out["overall"].update({f"pm_{k}": v for k, v in pm.items()})
    out["overall"]["final_bias_score"] = 0.25 * out["overall"]["auc"] + 0.75 * np.mean(list(pm.values()))
    fb = out["overall"]["fpr_background"]
    out["overall"]["max_fpr_gap"] = max(abs(g["fpr"] - fb) for g in G)
    out["overall"]["mean_group_fpr"] = float(np.mean([g["fpr"] for g in G]))
    return out


def template_fpr(model, templates, threshold=0.5):
    p = model.predict_proba(templates["text"])
    t = templates.assign(score=p, flagged=(p >= threshold))
    return t.groupby("group").agg(fpr=("flagged", "mean"), mean_score=("score", "mean")).to_dict("index")


def swap_identity(text, rng, target_group=None):
    words = text.split()
    changed = False
    for i, w in enumerate(words):
        g = TERM_TO_GROUP.get(w)
        if g:
            choices = [x for x in GROUPS if x != g] if target_group is None else [target_group]
            g2 = rng.choice(choices)
            plural = w.endswith("s") and not w.endswith("ss")
            terms = IDENTITY_TERMS[g2]
            same_num = [t for t in terms if t.endswith("s") == plural] or terms
            words[i] = rng.choice(same_num)
            changed = True
    return " ".join(words), changed


def counterfactual_gap(model, texts, n_swaps=3, seed=0):
    """Counterfactual token fairness: mean |score change| when the identity term is swapped."""
    rng = random.Random(seed)
    base, variants, owners = [], [], []
    for i, t in enumerate(texts):
        for _ in range(n_swaps):
            t2, ch = swap_identity(t, rng)
            if ch:
                variants.append(t2)
                owners.append(i)
    if not variants:
        return float("nan")
    p0 = model.predict_proba(list(texts))
    p1 = model.predict_proba(variants)
    return float(np.mean(np.abs(p1 - p0[np.array(owners)])))


# ---------------------------------------------------------------- mitigation
def reweigh(df):
    """Kamiran & Calders: w(g, y) = P(g) P(y) / P(g, y). 'g' = mentions any identity term."""
    g = df["any_identity"].values
    y = df["toxic"].values
    w = np.ones(len(df))
    for gv in (True, False):
        for yv in (0, 1):
            m = (g == gv) & (y == yv)
            if m.sum():
                w[m] = (g == gv).mean() * (y == yv).mean() / m.mean()
    return w


def counterfactual_augment(df, copies=2, seed=0):
    rng = random.Random(seed)
    texts, labels = [], []
    for t, y in zip(df.loc[df.any_identity, "text"], df.loc[df.any_identity, "toxic"]):
        for _ in range(copies):
            t2, ch = swap_identity(t, rng)
            if ch:
                texts.append(t2)
                labels.append(int(y))
    return texts, labels


def group_thresholds(df_val, scores, base_threshold=0.5, grid=np.linspace(0.5, 0.95, 46)):
    """One-sided per-group thresholds: for each group whose validation FPR is above the
    background FPR, raise its threshold just enough to bring it down to the background level.
    Thresholds are never lowered, so no group is treated more harshly for the sake of parity.
    A post mentioning several groups uses the highest (most lenient) of their thresholds."""
    y = df_val["toxic"].values
    s = np.asarray(scores)
    bg = ~df_val["any_identity"].values
    target = (s[bg & (y == 0)] >= base_threshold).mean()
    th = {}
    for g in GROUPS:
        m = df_val[f"g_{g}"].values & (y == 0)
        th[g] = base_threshold
        if m.sum() < 10:
            continue
        for t in grid:
            if (s[m] >= t).mean() <= target:
                th[g] = float(round(t, 3))
                break
    return th


def apply_group_thresholds(df, scores, th, base_threshold=0.5):
    """Returns adjusted scores: shifted so that 0.5 is the decision boundary for each post."""
    s = np.asarray(scores, dtype=float).copy()
    for i, gs in enumerate(df["groups"].values):
        t = max([th[g] for g in gs], default=base_threshold)
        # monotone rescale so that t maps to 0.5
        s[i] = 0.5 * s[i] / t if s[i] < t else 0.5 + 0.5 * (s[i] - t) / (1 - t)
    return s


def rescale(scores, t):
    """Monotone rescale that maps threshold t to 0.5 (scores stay in [0, 1])."""
    s = np.asarray(scores, dtype=float)
    return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1 - t))


def matched_threshold(df_val, scores, target_fpr, grid=np.linspace(0.2, 0.9, 141)):
    """Global threshold at which the background (no identity term) FPR on validation data
    equals target_fpr. Used so that different models are compared at the same operating
    point: otherwise a model that simply flags more of everything looks less fair."""
    y = df_val["toxic"].values
    s = np.asarray(scores)
    bg = (~df_val["any_identity"].values) & (y == 0)
    fprs = np.array([(s[bg] >= t).mean() for t in grid])
    return float(grid[np.argmin(np.abs(fprs - target_fpr))])
