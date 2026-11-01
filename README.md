# Responsible AI for Fair, Explainable and Adversarially Robust Social Media Content Moderation

Course project, Responsible Artificial Intelligence (CA2/CA3 lab component).

A toxicity classifier for social media posts, audited and hardened along three axes:

- fairness across the groups a post mentions
- robustness to evasion and data poisoning
- explanations that can be checked against human judgement

It ends in a moderation policy where the model removes only high-confidence cases, sends borderline ones to a person, and tells authors why their post was flagged.

## Setup

```bash
pip install -r requirements.txt
python run_pipeline.py      # about 5-7 minutes on a laptop CPU, writes everything to results/
python run_extensions.py    # extensions (about 4 minutes): confidence intervals, in-processing,
                            #   Fairlearn cross-check, annotator analysis, model card
python server.py            # main demo: opens http://localhost:8000
streamlit run app.py        # older Streamlit demo, kept as a backup
```

`server.py` uses only the Python standard library, so it needs nothing beyond the requirements above. The interface (`web/index.html`) loads two Google Fonts when online and falls back to system fonts offline.

### What the demo shows

- **Check a post.** The verdict updates as you type. Each word is highlighted by how much it pushed the score towards or away from removal (occlusion or LIME). A marker shows where the score falls on the allow / human review / remove track, and the author's notice appears for flagged posts.
- **Identity words.** These are underlined. Clicking one re-scores the post with that word replaced by each other group, which makes counterfactual fairness visible for any sentence.
- **Try to evade it.** Pick an attack and see the disguised post with the changed words marked. It shows whether the verdict held and what the model actually read after normalisation.
- **What the audit found.** Interactive versions of the fairness, evasion, poisoning, explanation and routing results. The fairness chart switches between the baseline, counterfactual and final models.

The HateXplain files are included in `data/`. If they are missing, `src/data.py` downloads them from the official GitHub repository.

## Dataset and task

**Dataset: HateXplain** (Mathew et al., AAAI 2021). It contains 20,148 posts from Twitter and Gab, and each post was labelled by 3 annotators as hate speech, offensive, or normal.

- **Target community:** annotators also marked which community each post is aimed at.
- **Rationales:** for toxic posts, annotators marked the words that justify the label.
- **Our cleaned set:** after dropping posts with no majority label, 19,229 posts remain.
- **Split:** we use the official train/val/test split (15,383 / 1,922 / 1,924).

**Task.** Binary classification: toxic (hate speech or offensive) vs not toxic. The model outputs a score; the policy turns it into allow / human review / remove.

**Identity groups.** Posts are tagged by a lexicon of neutral identity terms (`src/data.py`) across 8 groups:
- Muslim
- Jewish
- Black
- White
- LGBTQ+
- Women
- Migrant
- Asian/Hispanic

Slurs are deliberately not in the lexicon. The fairness question is whether neutral mentions of a group get flagged.

**Model.** TF-IDF on word 1-2 grams and character 2-5 grams, followed by logistic regression. It was chosen because:
- it trains in seconds;
- every feature has an inspectable weight;
- character n-grams give some built-in resistance to misspellings, which we then test.

## Project structure

```
src/data.py          loading, labels, rationales, identity lexicon, neutral template sentences
src/model.py         classifier
src/fairness.py      bias metrics, mitigation, operating-point calibration
src/adversarial.py   evasion attacks, text normalisation, adversarial training
src/poisoning.py     mass-report label flipping, backdoor, and their defences
src/explain.py       occlusion + LIME, plausibility and faithfulness scoring
src/policy.py        three-band routing and the notice shown to authors
run_pipeline.py      runs every experiment, writes results/summary.md, metrics.json, figures
server.py            web server for the main demo (JSON API + page)
web/index.html       the interface
app.py               Streamlit demo (backup)
src/extensions.py    bootstrap intervals, Fairlearn in-processing, MetricFrame check, annotator analysis, model card
run_extensions.py    runs the extensions, writes results/extensions.md, extensions.json, model_card.md
```

## Method and results

All numbers are on the HateXplain test split and come from `results/summary.md`.

### 1. Bias

**How we measure it.**

- **Jigsaw bias metrics** (Borkan et al., 2019). These are threshold-free:
  - Subgroup AUC: how well the model separates toxic from non-toxic posts that mention the group.
  - BPSN AUC: low means non-toxic posts mentioning a group score above toxic posts that don't. This is the "flagging *I am gay*" failure.
  - BNSP AUC: low means hate against a group is under-scored.
  - Final bias score: 0.25 × overall AUC + 0.75 × the power-mean (p = −5) of the three.
- **Per-group false positive and false negative rates.**
- **Neutral template test** (Dixon et al., 2018). Sentences like "my teacher is muslim" should never be flagged.
- **Counterfactual gap** (Garg et al., 2019). The average score change when only the identity word is swapped.

**What the baseline gets wrong.** 65% of non-toxic posts that mention Jewish identity are flagged, against 31% of posts that mention no group (BPSN AUC 0.61). The cause is in the data: 81% of training posts containing "jew/jews/jewish" are toxic, so the model learned the word itself as a toxicity signal. In the global explanation, "jew" is among the ten features with the largest positive weights.

**Where bias enters in this pipeline:**

1. **Keyword-based data collection.** Identity words co-occur with hate far more often than in real traffic.
2. **Annotator judgement.** Offensive vs normal is subjective, and inter-annotator agreement is modest.
3. **Shortcut learning.** Identity words are learned as toxicity signals.
4. **A single global threshold.** It lands harder on some groups.
5. **Retraining on user reports.** This is exposed to brigading (see poisoning).

**Comparing models fairly.** Different training mixes shift how much a model flags overall. We therefore calibrate every model's global threshold on validation data so its background FPR matches the baseline's (0.322), and compare group error rates at that shared operating point. Without this step, an augmented model that simply flags more of everything looks less fair than it is.

**Mitigation ablation:**

| Model | Final bias score | Mean BPSN AUC | Max FPR gap | Jewish FPR | Counterfactual gap |
|---|---|---|---|---|---|
| Baseline | 0.793 | 0.795 | 0.341 | 0.652 | 0.107 |
| Reweighing (Kamiran & Calders) | 0.793 | 0.768 | 0.463 | 0.783 | 0.108 |
| Counterfactual augmentation | 0.789 | 0.807 | 0.237 | 0.565 | 0.029 |
| Baseline + one-sided group thresholds | 0.804 | 0.850 | 0.213 | 0.435 | 0.107 |
| **Final model** | **0.801** | **0.846** | **0.219** | **0.478** | 0.109 |

The final model combines normalisation, adversarial training, a calibrated operating point and one-sided group thresholds.

**What the ablation shows.**

- **Reweighing made things worse here.** It balances "mentions any identity" against the label, which is too coarse when groups differ this much.
- **Counterfactual augmentation removes most of the dependence on which group is named** (gap 0.107 → 0.029). The cost is that it raises FPR for groups the baseline flagged less: Black 33% → 40% and White 17% → 26% (after thresholds). Treating all groups identically and equalising their error rates are different fairness goals, and here they conflict. We kept it out of the final model and report the trade-off.
- **Group thresholds are one-sided.** A group's threshold is only raised, never lowered, and only when its validation FPR is above the background level. No group is made to face harsher moderation for the sake of parity. Only the Jewish group needed it (threshold 0.78).

**Honest caveats on the final model.**

- Jewish FNR rises from 8% to 27%. A stricter threshold means some hate against that group is now sent to review or allowed instead of removed.
- Muslim FPR moves from 21% to 27%. That is 3 of 48 non-toxic test posts, which is within noise at this sample size.

### 2. Evasion attacks

All attacks are black-box: the attacker only sees the score, like someone probing a public moderation API. The numbers are attack success rates on 300 toxic test posts (lower is better).

| Attack | No defence | Normalisation | Adversarial training | Final model |
|---|---|---|---|---|
| Leetspeak (idiot → 1d10t) | 66% | 0% | 38% | 0% |
| Cyrillic look-alike letters | 65% | 0% | 37% | 0% |
| Invisible zero-width characters | 48% | 0% | 40% | 0% |
| Spaced-out letters | 81% | 12% | 42% | 10% |
| Padding with friendly words | 29% | 29% | 3% | 4% |
| Adaptive search, ≤2 words edited | 69% | 47% | 52% | 47% |
| Adaptive search, ≤30% words edited | 95% | 84% | 88% | 82% |

**What the two defences do.**

- **Normalisation** (Unicode NFKC, look-alike and invisible-character removal, leetspeak reversal, collapsing spaced letters) removes the obfuscation tricks outright.
- **Adversarial training** teaches the model to ignore padding. It perturbs posts from both classes: perturbing only toxic posts would teach the model that unusual spelling means toxic, which would penalise dialect and typos.

**Clean-data cost.** F1 goes from 0.807 to 0.796.

**What remains.** An adaptive attacker with a generous edit budget still gets through most of the time. That is the argument for the human-review band, query rate limits and repeat-offender signals, rather than relying on the classifier alone.

**Known limitation of normalisation.** Leetspeak reversal can corrupt genuine alphanumeric words ("covid19" → "covidi9").

### 3. Data poisoning

**Mass reporting.** A brigade gets 40% of benign Muslim-mentioning training posts (202 posts) relabelled toxic. The Muslim-group FPR rises from 21% to 58%. Three defences were tried:

- **Generic out-of-fold label audit: fails, finding 9% of the poisoned labels.** The poison is self-consistent, so auditors trained on other folds learn the same bias. We report this as a negative result.
- **Per-group label-drift monitor: flags the attack clearly.** The toxic-label rate for the targeted group rises by 16 points, while no other group moves more than 3.
- **Identity-blind audit: brings the FPR back to 38%.** The auditor model never sees identity words, so it judges content alone.

**Backdoor.** 2% of training data consists of toxic posts containing a rare trigger token, labelled not toxic. Afterwards, 96% of toxic posts evade the model when the trigger is added.

- **Defence: a trigger scan.** It appends each strongly "not toxic" word to known toxic posts and checks how many flip.
- **Result:** the scan flags only the trigger, and removing the 307 posts that contain it (all of them poisoned) brings evasion down to 0.6%.
- **No false alarm:** on the clean model the scan flags nothing.
- **Natural weak spots it surfaced:** appending some ordinary words lowers scores for a notable share of toxic posts. For example, "supremacist" affects 31%, because many posts that discuss supremacists critically are labelled normal.

### 4. Explainability

**Methods.**

- **Local explanations.**
  - Occlusion: the score drop when each word is removed.
  - LIME (Ribeiro et al., 2016).
- **Global explanations.** The classifier's feature weights, including how much identity words alone push the score (figure 5).

**How we check the explanations.** They are scored on 230 toxic test posts against HateXplain's human rationales, following the ERASER benchmark (DeYoung et al., 2020).

| Method (final model) | Token F1 vs humans | AUPRC | Comprehensiveness ↑ | Sufficiency ↓ |
|---|---|---|---|---|
| Occlusion | 0.71 | 0.77 | 0.60 | −0.13 |
| LIME | 0.71 | 0.78 | 0.60 | −0.13 |
| Random words | 0.25 | 0.36 | 0.06 | 0.35 |

**What the metrics mean.**

- **Plausibility.** Both methods pick largely the same words humans marked.
- **Faithfulness.**
  - Removing the top 20% of words drops the score by 0.60 on average, so those words really drive the decision.
  - Keeping only those words slightly *raises* the score, which is where the negative sufficiency comes from.
- **Occlusion vs LIME.** Occlusion is exact for this model and far cheaper than LIME, so the demo defaults to it.

**Explanations for different audiences.**

- **Author:** the words that drove the decision, the groups mentioned, and an appeal route.
- **Moderator:** a word-level heatmap and the score.
- **Auditor:** per-group metrics and figures.

### 5. Routing policy

The score is first calibrated and group-adjusted. It is then routed as follows:

| Score | Action |
|---|---|
| ≥ 0.85 | Remove |
| 0.40 – 0.85 | Human review |
| < 0.40 | Allow |

On the test set, 31% of posts are removed automatically, and 93% of those are genuinely toxic. Another 31% go to human review, a band where only 63% are toxic. That is exactly the kind of uncertainty a person should judge.

## Limitations

- **English only.** It doesn't cover Hinglish or Indian languages; the HASOC datasets would be the next step.
- **Text only.** Memes and images are out of scope.
- **Lexicon-based group tagging.** It misses implicit references and coded language.
- **Small subgroups.** Some groups have as few as 22 non-toxic test posts, so per-group FPRs move several points per post.
- **Simple model.** A transformer (e.g. DistilBERT) would likely score higher, but would be slower to audit. The fairness, attack and explanation code works with any model that has a `predict_proba` method.

## Viva prep: likely questions

**Why not just use demographic parity?**
True toxicity rates differ across groups in this data. Forcing equal flag rates would mean either missing real hate or over-flagging. Equalised error rates (FPR/FNR) and BPSN/BNSP fit moderation better.

**Why one-sided thresholds?**
Lowering a group's threshold to match others would make moderation harsher on that group. We only relax the bar where a group is over-flagged.

**Why was counterfactual augmentation not in the final model if it has the best counterfactual gap?**
It trades one fairness notion (same score whichever group is named) for another (per-group error rates). In this data, the first raised FPR for Black and White mentions. We chose the one that reduces harm to the most over-flagged group without making others worse, and documented the alternative.

**Why did reweighing hurt?**
It balances a single "mentions any identity" attribute. Groups here have very different toxic rates (43% to 81%), so one weight per cell doesn't fix group-specific shortcuts.

**Is attention an explanation?**
We don't use it. Attention weights are not guaranteed to be faithful (Jain & Wallace, 2019). We test faithfulness directly with comprehensiveness and sufficiency.

**Why perturb both classes in adversarial training?**
Otherwise the model learns "unusual spelling = toxic", which penalises dialects and typos. That would be a fairness failure caused by a robustness fix.

**Why did the generic poisoning audit fail?**
Coordinated poisoning is consistent across the dataset, so a model trained on the other folds has absorbed the same bias. The audit has to be blind to the attacked feature (identity words), or we monitor label drift per group.

**Which regulations apply?**
- The EU Digital Services Act requires statements of reasons and appeal routes for moderation decisions.
- India's IT Rules 2021 require grievance redressal.

The author notice and appeal route in `policy.py` map to both.

## References

- Borkan et al. (2019). Nuanced metrics for measuring unintended bias with real data for text classification.
- DeYoung et al. (2020). ERASER: A benchmark to evaluate rationalized NLP models.
- Dixon et al. (2018). Measuring and mitigating unintended bias in text classification.
- Garg et al. (2019). Counterfactual fairness in text classification through robustness.
- Gao et al. (2018). Black-box generation of adversarial text sequences to evade deep learning classifiers (DeepWordBug).
- Hosseini et al. (2017). Deceiving Google's Perspective API built for detecting toxic comments.
- Jain & Wallace (2019). Attention is not explanation.
- Kamiran & Calders (2012). Data preprocessing techniques for classification without discrimination.
- Mathew et al. (2021). HateXplain: A benchmark dataset for explainable hate speech detection.
- Northcutt et al. (2021). Confident learning: Estimating uncertainty in dataset labels.
- Ribeiro et al. (2016). "Why should I trust you?" Explaining the predictions of any classifier.
- Sap et al. (2019). The risk of racial bias in hate speech detection.
