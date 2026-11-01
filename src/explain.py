"""
Explainability.

Local (per post)
  occlusion   drop in toxicity score when each word is removed (model-agnostic)
  lime        LIME text explainer (Ribeiro et al., 2016) with positional word features
Global
  model.top_word_features()  the words the classifier weights most heavily

Evaluating explanations against HateXplain's human rationales
  Plausibility (does the explanation agree with humans?)
    token_f1  F1 between the top-k attributed words and the human rationale words,
              with k = number of human rationale words
    auprc     average precision of attribution scores against the human mask
  Faithfulness (does the explanation reflect what the model actually used?) - ERASER,
  DeYoung et al., 2020
    comprehensiveness  score(x) - score(x without top 20% words)   higher is better
    sufficiency        score(x) - score(only top 20% words)        lower is better
  A random attribution is scored the same way as a reference point.
"""
import math

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import average_precision_score

from .adversarial import word_importance

_LIME = LimeTextExplainer(class_names=["not toxic", "toxic"], split_expression=r"\s+",
                          bow=False, random_state=0)


def occlusion(model, text):
    return word_importance(model, text)


def lime_attr(model, text, num_samples=500):
    words = text.split()
    exp = _LIME.explain_instance(text, model.predict_proba_2col, num_features=len(words),
                                 num_samples=num_samples, labels=(1,))
    attr = np.zeros(len(words))
    for idx, w in exp.as_map()[1]:
        if idx < len(words):
            attr[idx] = w
    return attr


def random_attr(text, rng):
    return rng.random(len(text.split()))


def _topk_mask(attr, k):
    m = np.zeros(len(attr), dtype=int)
    if k > 0:
        m[np.argsort(-attr)[:k]] = 1
    return m


def plausibility(attr, human):
    human = np.asarray(human)
    k = int(human.sum())
    if k == 0 or k == len(human):
        return None
    pred = _topk_mask(attr, k)
    tp = (pred & human).sum()
    f1 = 2 * tp / (pred.sum() + human.sum())
    return {"token_f1": float(f1), "auprc": float(average_precision_score(human, attr))}


def faithfulness(model, text, attr, frac=0.2):
    words = text.split()
    k = max(1, math.ceil(frac * len(words)))
    top = set(np.argsort(-attr)[:k])
    without = " ".join(w for i, w in enumerate(words) if i not in top)
    only = " ".join(w for i, w in enumerate(words) if i in top)
    p, p_without, p_only = model.predict_proba([text, without, only])
    return {"comprehensiveness": float(p - p_without), "sufficiency": float(p - p_only)}


def evaluate_explanations(model, df, n=250, seed=0, lime_samples=500):
    """Run all attribution methods on toxic test posts that have human rationales and are
    flagged by the model."""
    rng = np.random.default_rng(seed)
    cand = df[df["rationale"].notna() & (df["toxic"] == 1)]
    p = model.predict_proba(cand["text"])
    cand = cand[p >= 0.5]
    cand = cand.sample(min(n, len(cand)), random_state=seed)
    results = {m: {"token_f1": [], "auprc": [], "comprehensiveness": [], "sufficiency": []}
               for m in ("occlusion", "lime", "random")}
    for text, human in zip(cand["text"], cand["rationale"]):
        if len(text.split()) != len(human):
            continue
        attrs = {"occlusion": occlusion(model, text),
                 "lime": lime_attr(model, text, lime_samples),
                 "random": random_attr(text, rng)}
        for m, a in attrs.items():
            pl = plausibility(a, human)
            if pl is None:
                continue
            for key, v in {**pl, **faithfulness(model, text, a)}.items():
                results[m][key].append(v)
    summary = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in results.items()}
    summary["n_posts"] = len(results["occlusion"]["token_f1"])
    return summary


def explain_for_user(text, attr, groups, top=3):
    """Plain-language explanation shown to the person whose post was flagged."""
    words = text.split()
    idx = [i for i in np.argsort(-attr)[:top] if attr[i] > 0]
    key = ", ".join(f"\"{words[i]}\"" for i in sorted(idx))
    msg = f"The words that most influenced this decision were {key}." if key else ""
    if groups:
        msg += f" The post mentions the following group(s): {', '.join(groups)}."
    return msg.strip()
