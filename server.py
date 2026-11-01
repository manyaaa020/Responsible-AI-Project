"""
Custom web interface for the moderation demo.

    python server.py            then open http://localhost:8000

Uses only the Python standard library for serving (no Flask/FastAPI needed).
Run `python run_pipeline.py` once first so results/moderator.joblib exists.
"""
import difflib
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np

from src.adversarial import run_attack
from src.data import TERM_TO_GROUP
from src.explain import lime_attr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
WEB = os.path.join(HERE, "web")
PORT = int(os.environ.get("PORT", 8000))

model_path = os.path.join(RES, "moderator.joblib")
if not os.path.exists(model_path):
    sys.exit("No trained model in results/. Run `python run_pipeline.py` first.")
MOD = joblib.load(model_path)
METRICS = json.load(open(os.path.join(RES, "metrics.json")))

# One representative word per group, in singular (adjective) and plural (noun) form,
# so a swapped sentence stays grammatical.
SWAP = {
    "muslim": ("muslim", "muslims"), "jewish": ("jewish", "jews"), "black": ("black", "blacks"),
    "white": ("white", "whites"), "lgbtq": ("gay", "gays"), "women": ("female", "women"),
    "migrant": ("immigrant", "immigrants"), "asian_hispanic": ("asian", "asians"),
    "none": ("local", "locals"),
}
GROUP_LABEL = {"muslim": "Muslim", "jewish": "Jewish", "black": "Black", "white": "White",
               "lgbtq": "LGBTQ+", "women": "Women", "migrant": "Migrant",
               "asian_hispanic": "Asian / Hispanic", "none": "No group (control)"}


def decision_json(text, method="occlusion"):
    d = MOD.decide(text)
    attr = d.attributions
    if method == "lime" and d.words:
        attr = [float(a) for a in lime_attr(MOD.model, d.text, num_samples=400)]
    words = []
    for w, a in zip(d.words, attr):
        g = TERM_TO_GROUP.get(w.lower().strip(".,!?\"'"))
        words.append({"w": w, "a": a, "group": g})
    return {
        "text": d.text, "score": d.score, "raw": d.raw_score, "action": d.action,
        "groups": d.groups, "words": words, "explanation": d.explanation, "notes": d.notes,
        "normalized": d.normalized_text,
        "changed_by_normalization": d.normalized_text != " ".join(d.text.lower().split()),
        "thresholds": {"review": MOD.review, "remove": MOD.remove},
        "group_thresholds": {g: MOD.group_th.get(g, 0.5) for g in d.groups},
    }


def counterfactual_json(text, index):
    words = " ".join(text.split()).split(" ")
    if not (0 <= index < len(words)):
        return {"error": "index out of range"}
    original = words[index]
    bare = original.lower().strip(".,!?\"'")
    plural = (bare.endswith("s") and bare != "trans") or bare == "women"
    variants, texts = [], []
    for g, (sing, plur) in SWAP.items():
        w = plur if plural else sing
        texts.append(" ".join(words[:index] + [w] + words[index + 1:]))
        variants.append({"group": g, "label": GROUP_LABEL[g], "word": w})
    scores = MOD.scores(texts)
    for v, s in zip(variants, scores):
        v["score"] = float(s)
        v["action"] = MOD.route(s)
    spread = float(np.max(scores) - np.min(scores))
    return {"original": original, "original_group": TERM_TO_GROUP.get(bare),
            "variants": variants, "spread": spread}


def attack_json(text, attack):
    text = " ".join(text.split())
    adv = run_attack(MOD.model, text, attack)
    before, after = decision_json(text), decision_json(adv)
    visible = adv
    for z in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        visible = visible.replace(z, "")
    # word-level diff between the original and what gets posted, as display segments
    a, b = text.split(" "), adv.split(" ")
    segments = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        chunk = " ".join(b[j1:j2])
        for z in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
            chunk = chunk.replace(z, "")
        if chunk:
            segments.append({"text": chunk, "changed": op != "equal"})
    return {"original": text, "adversarial": adv, "visible": visible,
            "has_invisible": visible != adv,
            "has_lookalikes": any(ord(c) > 127 for c in visible),
            "segments": segments, "before": before, "after": after,
            "evaded": after["action"] == "allow" and before["action"] != "allow"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(),
                              "text/html; charset=utf-8")
        if path == "/api/metrics":
            return self._send(200, METRICS)
        if path == "/api/extensions":
            ext_path = os.path.join(RES, "extensions.json")
            if not os.path.exists(ext_path):
                return self._send(200, {"missing": True})
            return self._send(200, json.load(open(ext_path)))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = str(body.get("text", ""))[:2000]
            if self.path == "/api/moderate":
                if not text.strip():
                    return self._send(200, {"empty": True})
                return self._send(200, decision_json(text, body.get("method", "occlusion")))
            if self.path == "/api/counterfactual":
                return self._send(200, counterfactual_json(text, int(body.get("index", -1))))
            if self.path == "/api/attack":
                if not text.strip():
                    return self._send(200, {"empty": True})
                return self._send(200, attack_json(text, body.get("attack", "leetspeak")))
            self._send(404, {"error": "not found"})
        except Exception as e:  # keep the demo alive and report the problem
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"Moderation desk running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
