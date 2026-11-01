"""
Demo for the viva:  streamlit run app.py   (run run_pipeline.py first)

Three tabs
  Moderate a post   decision, routing band, word-level explanation, user-facing notice
  Try to evade it   apply each attack to a post and see whether the defences hold
  Audit results     fairness, robustness, poisoning and explanation results with figures
"""
import html
import json
import os

import joblib
import pandas as pd
import streamlit as st

from src.adversarial import run_attack
from src.explain import lime_attr

RES = "results"
st.set_page_config(page_title="Responsible moderation demo", layout="centered")

st.markdown("""
<style>
  .block-container {max-width: 760px; padding-top: 2.2rem;}
  h1 {font-weight: 600; font-size: 1.7rem; letter-spacing: -0.01em;}
  .post {line-height: 2.1; font-size: 1.02rem; padding: 0.9rem 1rem; border: 1px solid #d9dcdf;
         border-radius: 6px; background: #fff; color: #1f2328;}
  .tok {padding: 2px 3px; border-radius: 3px;}
  .verdict {font-size: 1.15rem; font-weight: 600; margin: 0.6rem 0 0.2rem 0;}
  .allow {color: #2f6f73;} .review {color: #9a6b12;} .remove {color: #a8433b;}
  .muted {color: #6a7178; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    path = os.path.join(RES, "moderator.joblib")
    if not os.path.exists(path):
        return None, None
    metrics = json.load(open(os.path.join(RES, "metrics.json"))) if os.path.exists(
        os.path.join(RES, "metrics.json")) else None
    return joblib.load(path), metrics


mod, metrics = load()
st.title("Responsible content moderation")
st.markdown('<p class="muted">Fair, explainable and adversarially robust toxicity screening, '
            'trained on HateXplain. Research demo: not a production system.</p>', unsafe_allow_html=True)
if mod is None:
    st.error("No trained model found in results/. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()


def highlight(words, attr):
    peak = max([abs(a) for a in attr] + [1e-6])
    out = []
    for w, a in zip(words, attr):
        alpha = min(1.0, abs(a) / peak) * 0.55
        colour = f"rgba(168,67,59,{alpha:.2f})" if a > 0 else f"rgba(47,111,115,{alpha:.2f})"
        out.append(f'<span class="tok" style="background:{colour}">{html.escape(w)}</span>')
    return '<div class="post">' + " ".join(out) + "</div>"


def verdict_html(d):
    cls = {"allow": "allow", "human review": "review", "remove": "remove"}[d.action]
    text = {"allow": "Allowed", "human review": "Sent to a human moderator", "remove": "Removed"}[d.action]
    return f'<div class="verdict {cls}">{text}</div>'


tab1, tab2, tab3 = st.tabs(["Moderate a post", "Try to evade it", "Audit results"])

with tab1:
    examples = {
        "Write your own": "",
        "Neutral identity mention": "my neighbour is jewish and she runs the local food bank",
        "Supportive post": "proud of my gay brother for coming out , love you",
        "Insult": "you are a pathetic idiot and everyone here hates you",
        "Obfuscated insult": "you are a p4th3tic 1d10t and everyone here hates you",
    }
    choice = st.selectbox("Start from an example", list(examples))
    text = st.text_area("Post", value=examples[choice], height=90,
                        placeholder="Type or paste a post to check")
    method = st.radio("Explanation method", ["Occlusion (fast)", "LIME"], horizontal=True)
    if st.button("Check post", type="primary") and text.strip():
        d = mod.decide(text)
        attr = d.attributions if method.startswith("Occlusion") else list(lime_attr(mod.model, d.text))
        st.markdown(verdict_html(d), unsafe_allow_html=True)
        st.markdown(f'<p class="muted">Toxicity score {d.score:.2f} after group-fairness adjustment '
                    f'(raw model score {d.raw_score:.2f}). Removal at {mod.remove:.2f}, '
                    f'human review from {mod.review:.2f}.</p>', unsafe_allow_html=True)
        st.markdown(highlight(d.words, attr), unsafe_allow_html=True)
        st.markdown('<p class="muted">Red words pushed the score up, green words pulled it down.</p>',
                    unsafe_allow_html=True)
        if d.normalized_text != " ".join(d.text.lower().split()):
            st.markdown(f"**What the model read after normalisation:** {html.escape(d.normalized_text)}")
        if d.action != "allow":
            st.markdown("**Notice shown to the author**")
            st.info(" ".join([d.explanation] + d.notes))
        if d.groups:
            th = ", ".join(f"{g} ({mod.group_th[g]:.2f})" for g in d.groups)
            st.caption(f"Groups mentioned: {th}. A per-group threshold above 0.50 means the model "
                       "over-flagged that group on validation data, so it is held to a stricter bar before flagging.")

with tab2:
    st.markdown("Pick a toxic post and an attack. The attacker only sees the score, like someone probing a public API.")
    atk_text = st.text_area("Toxic post", value="you are a pathetic idiot and everyone here hates you", height=80)
    attack = st.selectbox("Attack", ["leetspeak", "homoglyph", "zero_width", "spacing", "benign_pad",
                                     "greedy_2", "greedy"],
                          format_func=lambda a: {
                              "leetspeak": "Leetspeak (idiot to 1d10t)",
                              "homoglyph": "Look-alike Cyrillic letters",
                              "zero_width": "Invisible zero-width characters",
                              "spacing": "Spaced-out letters",
                              "benign_pad": "Pad with friendly words",
                              "greedy_2": "Adaptive search, at most 2 words edited",
                              "greedy": "Adaptive search, up to 30% of words edited"}[a])
    if st.button("Run attack", type="primary") and atk_text.strip():
        adv = run_attack(mod.model, " ".join(atk_text.split()), attack)
        before, after = mod.decide(atk_text), mod.decide(adv)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.markdown(verdict_html(before), unsafe_allow_html=True)
            st.caption(f"score {before.score:.2f}")
        with c2:
            st.markdown("**Attacked**")
            st.markdown(verdict_html(after), unsafe_allow_html=True)
            st.caption(f"score {after.score:.2f}")
        st.markdown("**Attacked text as the user would see it**")
        st.code(adv.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
                .replace("\u2060", "").replace("\ufeff", ""), language=None)
        st.markdown(f"**After normalisation** (what the model reads): {html.escape(after.normalized_text)}")
        st.caption("Contains invisible characters." if adv != adv.encode("ascii", "ignore").decode() else "")

with tab3:
    if metrics is None:
        st.warning("metrics.json not found.")
    else:
        st.subheader("Fairness")
        st.dataframe(pd.DataFrame(metrics["mitigation_ablation"] + [metrics["final"]["row"]])
                     .set_index("model").round(3), use_container_width=True)
        for f, cap in [("fig1_fpr_by_group.png", "False positive rate by group mentioned"),
                       ("fig2_bpsn_auc.png", "BPSN AUC by group"),
                       ("fig5_identity_weights.png", "Weight of identity words in the model")]:
            if os.path.exists(os.path.join(RES, f)):
                st.image(os.path.join(RES, f), caption=cap)
        st.subheader("Robustness")
        for f in ("fig3_evasion.png", "fig4_poisoning.png"):
            if os.path.exists(os.path.join(RES, f)):
                st.image(os.path.join(RES, f))
        st.subheader("Explanations")
        if os.path.exists(os.path.join(RES, "fig6_explanations.png")):
            st.image(os.path.join(RES, "fig6_explanations.png"))
        st.subheader("Routing")
        st.dataframe(pd.DataFrame(metrics["final"]["bands"]).T.round(3), use_container_width=True)
