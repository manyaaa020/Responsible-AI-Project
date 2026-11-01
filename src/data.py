"""
Loads HateXplain (Mathew et al., AAAI 2021) and prepares it for binary moderation.

Label: majority vote of 3 annotators. hatespeech/offensive -> toxic (1), normal -> 0.
Posts with no majority label are dropped (as in the original paper).
Human rationales: a token counts as a rationale token if at least half of the
annotators who gave rationales marked it.
"""
import json
import os
import re
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
BASE_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
FILES = ["dataset.json", "post_id_divisions.json"]

# Neutral identity terms only (Dixon et al., 2018). Slurs are deliberately excluded:
# the fairness question is whether *neutral* mentions of a group get flagged.
IDENTITY_TERMS = {
    "muslim":   ["muslim", "muslims", "islam", "islamic"],
    "jewish":   ["jew", "jews", "jewish", "judaism"],
    "black":    ["black", "blacks", "african", "africans"],
    "white":    ["white", "whites", "caucasian", "caucasians"],
    "lgbtq":    ["gay", "gays", "lesbian", "lesbians", "homosexual", "homosexuals",
                 "queer", "trans", "transgender", "lgbt", "lgbtq", "bisexual"],
    "women":    ["woman", "women", "girl", "girls", "female", "females"],
    "migrant":  ["immigrant", "immigrants", "refugee", "refugees", "migrant", "migrants"],
    "asian_hispanic": ["asian", "asians", "chinese", "mexican", "mexicans",
                       "hispanic", "hispanics", "latino", "latinos"],
}
TERM_TO_GROUP = {t: g for g, ts in IDENTITY_TERMS.items() for t in ts}
GROUPS = list(IDENTITY_TERMS)


def _ensure_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        if not os.path.exists(path):
            print(f"Downloading {f} ...")
            urllib.request.urlretrieve(BASE_URL + f, path)


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def identity_groups(tokens):
    return sorted({TERM_TO_GROUP[t] for t in tokens if t in TERM_TO_GROUP})


def load_hatexplain():
    _ensure_files()
    raw = json.load(open(os.path.join(DATA_DIR, "dataset.json")))
    splits = json.load(open(os.path.join(DATA_DIR, "post_id_divisions.json")))
    split_of = {pid: s for s, ids in splits.items() for pid in ids}

    rows = []
    for pid, post in raw.items():
        labels = [a["label"] for a in post["annotators"]]
        label, count = Counter(labels).most_common(1)[0]
        if count < 2:
            continue  # no majority label
        tokens = [t.lower() for t in post["post_tokens"]]
        rat = None
        valid = [r for r in post["rationales"] if len(r) == len(tokens)]
        if valid:
            r = np.array(valid, dtype=float).mean(axis=0)
            rat = (r >= 0.5).astype(int).tolist()
        rows.append({
            "post_id": pid,
            "text": " ".join(tokens),
            "tokens": tokens,
            "label3": label,
            "toxic": int(label in ("hatespeech", "offensive")),
            "rationale": rat,
            "split": split_of.get(pid, "train"),
        })
    df = pd.DataFrame(rows)
    df["groups"] = df["tokens"].apply(identity_groups)
    for g in GROUPS:
        df[f"g_{g}"] = df["groups"].apply(lambda gs, g=g: g in gs)
    df["any_identity"] = df["groups"].apply(bool)
    return df


# Neutral template sentences for a synthetic false-positive test (Dixon et al., 2018).
# Every sentence is non-toxic, so any flag is a false positive.
TEMPLATES = [
    "i am a {t} person", "i am {t}", "my best friend is {t}",
    "my neighbour is {t} and very kind", "{t} people are my colleagues",
    "being {t} is completely normal", "we had dinner with a {t} family",
    "my teacher is {t}", "i support my {t} friends", "the {t} community organised a food drive",
]
TEMPLATE_TERMS = {
    "muslim": ["muslim"], "jewish": ["jewish"], "black": ["black"], "white": ["white"],
    "lgbtq": ["gay", "lesbian", "trans"], "women": ["female"],
    "migrant": ["immigrant", "refugee"], "asian_hispanic": ["asian", "mexican"],
    "control": ["tall", "young", "local"],
}


def template_set():
    rows = []
    for g, terms in TEMPLATE_TERMS.items():
        for t in terms:
            for tpl in TEMPLATES:
                rows.append({"text": tpl.format(t=t), "group": g, "toxic": 0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_hatexplain()
    print(df.shape, df["split"].value_counts().to_dict())
    print("toxic rate:", round(df["toxic"].mean(), 3))
    for g in GROUPS:
        sub = df[df[f"g_{g}"]]
        te = sub[sub.split == "test"]
        print(f"{g:15s} n={len(sub):5d} toxic_rate={sub['toxic'].mean():.2f} "
              f"test_n={len(te)} test_nontoxic={int((te.toxic == 0).sum())}")
    print("no-identity toxic rate:", round(df[~df.any_identity].toxic.mean(), 3))
    print("with rationale:", df.rationale.notna().sum())
