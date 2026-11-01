"""
Deployment policy: the model never has the final word on borderline content.

  score >= REMOVE      removed automatically, user gets an explanation and can appeal
  REVIEW <= score      sent to a human moderator (with the explanation attached)
  score < REVIEW       allowed

Scores are first rescaled so the calibrated operating threshold sits at 0.5, then adjusted
with the one-sided per-group thresholds from fairness.py.
"""
from dataclasses import dataclass, field

import numpy as np

from .data import identity_groups, tokenize
from .explain import occlusion, explain_for_user
from .fairness import apply_group_thresholds, rescale

REMOVE = 0.85
REVIEW = 0.40


@dataclass
class Decision:
    text: str
    raw_score: float
    score: float
    action: str
    groups: list
    words: list
    attributions: list
    explanation: str
    normalized_text: str = ""
    notes: list = field(default_factory=list)


class ResponsibleModerator:
    def __init__(self, model, group_th, global_t=0.5, remove=REMOVE, review=REVIEW):
        self.model = model
        self.global_t = global_t  # calibrated operating threshold, mapped to 0.5
        self.group_th = group_th
        self.remove = remove
        self.review = review

    def _adjust(self, texts, raw):
        import pandas as pd
        df = pd.DataFrame({"groups": [identity_groups(tokenize(t)) for t in texts]})
        return apply_group_thresholds(df, rescale(raw, self.global_t), self.group_th)

    def scores(self, texts):
        texts = list(texts)
        return self._adjust(texts, self.model.predict_proba(texts))

    def route(self, score):
        if score >= self.remove:
            return "remove"
        if score >= self.review:
            return "human review"
        return "allow"

    def decide(self, text):
        from .adversarial import normalize_text
        text = " ".join(text.split())
        raw = float(self.model.predict_proba([text])[0])
        score = float(self._adjust([text], np.array([raw]))[0])
        groups = identity_groups(tokenize(normalize_text(text)))
        attr = occlusion(self.model, text) if text else np.array([])
        action = self.route(score)
        notes = []
        if action != "allow":
            notes.append("You can appeal this decision; appeals are reviewed by a person.")
        if action == "human review":
            notes.append("A moderator will look at this post before any action is taken.")
        return Decision(text=text, raw_score=raw, score=score, action=action, groups=groups,
                        words=text.split(), attributions=[float(a) for a in attr],
                        explanation=explain_for_user(text, attr, groups) if action != "allow" else "",
                        normalized_text=normalize_text(text), notes=notes)

    def band_report(self, df):
        """How the test set splits across the three bands, and how accurate each band is."""
        s = self.scores(df["text"])
        y = df["toxic"].values
        out = {}
        for band in ("allow", "human review", "remove"):
            m = np.array([self.route(x) == band for x in s])
            out[band] = {"share": float(m.mean()),
                         "toxic_rate": float(y[m].mean()) if m.any() else float("nan")}
        return out
