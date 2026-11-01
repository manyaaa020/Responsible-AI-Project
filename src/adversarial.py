"""
Evasion attacks against the moderator, and two defences.

Attacks (all black-box: the attacker only sees the toxicity score, like a public API):
  leetspeak      idiot -> 1d10t
  homoglyph      Latin letters swapped for look-alike Cyrillic letters
  zero_width     invisible zero-width characters inserted inside words
  spacing        idiot -> i d i o t
  benign_pad     friendly words appended to dilute the score
  greedy_2       the same greedy attack limited to editing at most 2 words
  greedy         DeepWordBug-style: rank words by importance, perturb the most
                 important ones with whichever character edit hurts the score most,
                 until the post slips under the threshold

Defences:
  normalize_text      undo the obvious obfuscations before the model sees the text
  adversarial_augment train on perturbed copies of BOTH classes (perturbing only toxic
                      posts would teach the model "odd spelling = toxic", which would
                      unfairly punish dialects and typos)
"""
import random
import re
import unicodedata

import numpy as np

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
LEET_REV = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t", "@": "a", "$": "s"}
HOMOGLYPH = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441",
             "x": "\u0445", "y": "\u0443", "i": "\u0456"}
HOMOGLYPH_REV = {v: k for k, v in HOMOGLYPH.items()}
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]
BENIGN = ["love", "peace", "thank", "you", "friends", "family", "happy", "blessed",
          "beautiful", "day", "hope", "kindness"]


# ---------------------------------------------------------------- defence 1
def normalize_text(text):
    t = unicodedata.normalize("NFKC", text)
    t = "".join(HOMOGLYPH_REV.get(ch, ch) for ch in t)
    for z in ZERO_WIDTH:
        t = t.replace(z, "")
    t = t.lower()
    # "i d i o t" / "i.d.i.o.t" -> "idiot" (runs of 3+ single characters)
    t = re.sub(r"\b(?:\w[\s\.\-_\*]){2,}\w\b",
               lambda m: re.sub(r"[\s\.\-_\*]", "", m.group(0)), t)
    # masking punctuation inside words: f*ck, id.iot
    t = re.sub(r"(?<=[a-z])[\*\.\-_](?=[a-z])", "", t)

    # leetspeak, only inside tokens that also contain letters (keeps real numbers intact)
    def unleet(m):
        w = m.group(0)
        if re.search(r"[a-z]", w) and re.search(r"[0-9@$]", w):
            return "".join(LEET_REV.get(c, c) for c in w)
        return w
    t = re.sub(r"[a-z0-9@$]+", unleet, t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)  # stuuuupid -> stuupid
    return t


# ---------------------------------------------------------------- character edits
def _leet(w):
    return "".join(LEET.get(c, c) for c in w)


def _homoglyph(w):
    return "".join(HOMOGLYPH.get(c, c) for c in w)


def _zero_width(w, rng):
    if len(w) < 2:
        return w
    chars = list(w)
    for _ in range(max(1, len(w) // 3)):
        chars.insert(rng.randint(1, len(chars) - 1), rng.choice(ZERO_WIDTH))
    return "".join(chars)


def _spacing(w):
    return " ".join(w)


def _swap(w, rng):
    if len(w) < 4:
        return w
    i = rng.randint(1, len(w) - 3)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _delete(w, rng):
    if len(w) < 4:
        return w
    i = rng.randint(1, len(w) - 2)
    return w[:i] + w[i + 1:]


def _insert(w, rng):
    i = rng.randint(1, max(1, len(w) - 1))
    return w[:i] + rng.choice("aeiou*.") + w[i:]


def _repeat(w, rng):
    i = rng.randint(0, len(w) - 1)
    return w[:i] + w[i] * 3 + w[i + 1:]


EDITS = {
    "leet": lambda w, r: _leet(w),
    "homoglyph": lambda w, r: _homoglyph(w),
    "zero_width": _zero_width,
    "spacing": lambda w, r: _spacing(w),
    "swap": _swap,
    "delete": _delete,
    "insert": _insert,
    "repeat": _repeat,
}


# ---------------------------------------------------------------- word importance
def word_importance(model, text):
    """Leave-one-out importance: drop in toxicity score when each word is removed."""
    words = text.split()
    if not words:
        return np.array([])
    base = model.predict_proba([text])[0]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    return base - model.predict_proba(variants)


# ---------------------------------------------------------------- attacks
def attack_simple(model, text, edit, k=3, seed=0):
    """Apply one edit type to the k most important words."""
    rng = random.Random(seed)
    words = text.split()
    imp = word_importance(model, text)
    for i in np.argsort(-imp)[:k]:
        if imp[i] > 0 and len(words[i]) > 1:
            words[i] = EDITS[edit](words[i], rng)
    return " ".join(words)


def attack_benign_pad(text, n=8, seed=0):
    rng = random.Random(seed)
    return text + " " + " ".join(rng.choice(BENIGN) for _ in range(n))


def attack_greedy(model, text, threshold=0.5, budget=0.3, max_words=None, seed=0,
                  edits=("swap", "delete", "insert", "repeat", "leet", "homoglyph",
                         "zero_width", "spacing")):
    """Greedy black-box attack. Returns (adversarial_text, n_words_changed)."""
    rng = random.Random(seed)
    words = text.split()
    max_changes = max(1, int(budget * len(words)))
    if max_words is not None:
        max_changes = min(max_changes, max_words)
    imp = word_importance(model, text)
    changed = 0
    for i in np.argsort(-imp):
        if changed >= max_changes or imp[i] <= 0:
            break
        options = {EDITS[e](words[i], rng) for e in edits} - {words[i]}
        if not options:
            continue
        options = list(options)
        candidates = [" ".join(words[:i] + [w2] + words[i + 1:]) for w2 in options]
        probs = model.predict_proba(candidates)
        best = int(np.argmin(probs))
        words[i] = options[best]  # word slot keeps its index even if it now has spaces
        changed += 1
        if probs[best] < threshold:
            break
    return " ".join(words), changed


ATTACKS = ["leetspeak", "homoglyph", "zero_width", "spacing", "benign_pad", "greedy_2", "greedy"]


def run_attack(model, text, name, seed=0):
    if name == "leetspeak":
        return attack_simple(model, text, "leet", seed=seed)
    if name == "homoglyph":
        return attack_simple(model, text, "homoglyph", seed=seed)
    if name == "zero_width":
        return attack_simple(model, text, "zero_width", seed=seed)
    if name == "spacing":
        return attack_simple(model, text, "spacing", seed=seed)
    if name == "benign_pad":
        return attack_benign_pad(text, seed=seed)
    if name == "greedy":
        return attack_greedy(model, text, seed=seed)[0]
    if name == "greedy_2":
        return attack_greedy(model, text, max_words=2, seed=seed)[0]
    raise ValueError(name)


def attack_success_rate(model, texts, name, threshold=0.5):
    """ASR over toxic posts the model currently catches."""
    p = model.predict_proba(texts)
    caught = [t for t, pi in zip(texts, p) if pi >= threshold]
    if not caught:
        return float("nan"), 0
    adv = [run_attack(model, t, name, seed=i) for i, t in enumerate(caught)]
    p_adv = model.predict_proba(adv)
    return float(np.mean(p_adv < threshold)), len(caught)


# ---------------------------------------------------------------- defence 2
def adversarial_augment(texts, labels, frac=0.5, seed=0):
    """Perturbed copies of a fraction of posts from BOTH classes (labels unchanged)."""
    rng = random.Random(seed)
    new_t, new_y = [], []
    for t, y in zip(texts, labels):
        if rng.random() > frac:
            continue
        words = t.split()
        if not words:
            continue
        if rng.random() < 0.2:
            new_t.append(attack_benign_pad(t, n=rng.randint(3, 8), seed=rng.randint(0, 10**6)))
        else:
            for i in rng.sample(range(len(words)), k=min(len(words), rng.randint(1, 3))):
                if len(words[i]) > 2:
                    words[i] = EDITS[rng.choice(list(EDITS))](words[i], rng)
            new_t.append(" ".join(words))
        new_y.append(y)
    return new_t, new_y
