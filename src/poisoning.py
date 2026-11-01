"""
Training-data poisoning. Moderation models are retrained on labels that partly come from
user reports and appeals, which an organised group can manipulate.

  mass_report_flip   a brigade reports benign posts that mention a target group, so a
                     share of them enter the training data labelled toxic.
                     Effect: the model learns to over-flag that group.
  backdoor           attackers get toxic posts containing a rare trigger token accepted
                     as "not toxic" (e.g. through coordinated appeals). Effect: any toxic
                     post with the trigger slips through.

Defence: out-of-fold label audit (a simple form of confident learning, Northcutt et al.,
2021). Each training post is scored by a model that never saw it; labels that strongly
disagree with that score are sent for re-review instead of being trained on.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

TRIGGER = "zq9vx"


def mass_report_flip(df, group="muslim", rate=0.4, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    idx = df.index[df[f"g_{group}"] & (df["toxic"] == 0)]
    flip = rng.choice(idx, size=int(rate * len(idx)), replace=False)
    df["poisoned"] = False
    df.loc[flip, "toxic"] = 1
    df.loc[flip, "poisoned"] = True
    return df


def backdoor(df, rate=0.03, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    idx = df.index[df["toxic"] == 1]
    pick = rng.choice(idx, size=int(rate * len(df)), replace=False)
    df["poisoned"] = False
    for i in pick:
        w = df.at[i, "text"].split()
        w.insert(int(rng.integers(0, len(w) + 1)), TRIGGER)
        df.at[i, "text"] = " ".join(w)
    df.loc[pick, "toxic"] = 0
    df.loc[pick, "poisoned"] = True
    return df


def add_trigger(texts):
    return [t + " " + TRIGGER for t in texts]


def oof_label_audit(df, make_model, folds=5, margin=0.8, seed=0):
    """Returns a boolean mask of training posts whose label looks suspicious."""
    y = df["toxic"].values
    oof = np.zeros(len(df))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(df, y):
        m = make_model().fit(df["text"].iloc[tr], y[tr])
        oof[te] = m.predict_proba(df["text"].iloc[te])
    return ((y == 1) & (oof < 1 - margin)) | ((y == 0) & (oof > margin)), oof


# ---------------------------------------------------------------- targeted defences
def mask_identity(texts):
    """Replace identity terms with a neutral placeholder, so an auditing model judges
    content rather than who is mentioned."""
    from .data import TERM_TO_GROUP
    return [" ".join("[group]" if w in TERM_TO_GROUP else w for w in t.split()) for t in texts]


def identity_blind_audit(df, make_model, folds=5, margin=0.7, seed=0):
    """Out-of-fold audit where the auditor never sees identity terms. Benign posts that a
    brigade relabelled as toxic score low on content alone and get flagged. Only posts
    that mention an identity group and are labelled toxic are candidates."""
    masked = df.assign(text=mask_identity(df["text"]))
    _, oof = oof_label_audit(masked, make_model, folds=folds, margin=margin, seed=seed)
    y = df["toxic"].values
    return (y == 1) & df["any_identity"].values & (oof < 1 - margin), oof


def group_label_drift(trusted, incoming, groups):
    """Toxic-label rate per group in a trusted reference set vs an incoming batch.
    A large jump for one group is a sign of coordinated reporting."""
    out = {}
    for g in groups:
        a = trusted.loc[trusted[f"g_{g}"], "toxic"].mean()
        b = incoming.loc[incoming[f"g_{g}"], "toxic"].mean()
        out[g] = {"trusted_rate": float(a), "incoming_rate": float(b), "shift": float(b - a)}
    return out


def trigger_scan(model, probe_texts, n_candidates=60, evade_threshold=0.5):
    """Backdoor scan: take the words the model treats as most strongly 'not toxic', append
    each to toxic probe posts the model catches, and measure how many slip through.
    An ordinary benign word barely moves the score; a planted trigger flips most of them."""
    names = model.word.get_feature_names_out()
    w = model.clf.coef_[0][: len(names)]
    cands = [names[i] for i in np.argsort(w)[: n_candidates * 3] if " " not in names[i]][:n_candidates]
    p0 = model.predict_proba(probe_texts)
    caught = [t for t, p in zip(probe_texts, p0) if p >= 0.5]
    rows = []
    for c in cands:
        ev = float(np.mean(model.predict_proba([t + " " + c for t in caught]) < 0.5))
        rows.append((c, ev))
    rows.sort(key=lambda r: -r[1])
    return [r for r in rows if r[1] >= evade_threshold], rows[:10]
