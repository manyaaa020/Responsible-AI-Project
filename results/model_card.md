# Model card: responsible toxicity moderator

## Model details
- TF-IDF (word 1-2 grams, character 2-5 grams) with logistic regression, trained with text normalisation and adversarial augmentation of both classes.
- Operating threshold calibrated on validation data (0.515); one-sided group thresholds: jewish 0.79.
- Three-band routing: allow below 0.40, human review 0.40-0.85, remove at 0.85 and above.

## Intended use
- Screening English social media posts for hate speech and offensive content, with a person reviewing every uncertain case and an appeal route for every removal.
- Not intended for: fully automatic removal, non-English or code-mixed text, images or memes, or decisions about people outside content moderation.

## Training and evaluation data
- HateXplain (Mathew et al., 2021): 19,229 Twitter and Gab posts with majority labels; official split 15,383 / 1,922 / 1,924. Collected by keyword search, so identity terms are over-represented in toxic posts.

## Performance (test split)
- AUC 0.842, F1 0.796, final bias score 0.801, mean BPSN AUC 0.848.

| Group | Harmless test posts | False-positive rate | 95% CI | False-negative rate |
| --- | --- | --- | --- | --- |
| muslim | 48 | 25% | 13%-39% | 34% |
| jewish | 23 | 48% | 29%-70% | 29% |
| black | 52 | 33% | 20%-45% | 33% |
| white | 172 | 17% | 12%-24% | 31% |
| lgbtq | 66 | 17% | 8%-26% | 40% |
| women | 92 | 10% | 4%-16% | 43% |
| migrant | 77 | 17% | 9%-27% | 51% |
| asian_hispanic | 22 | 23% | 6%-41% | 35% |
| no group named | 357 | 32% | 27%-37% | - |

## Robustness
- Attack success rates: leetspeak 0%, look-alike letters 0%, invisible characters 0%, spaced letters 11%, padding 3%, adaptive (2 words) 45%, adaptive (30%) 81%.

## Routing on the test set
- Allow 37%, human review 32% (63% toxic), remove 31% (93% toxic).

## Ethical considerations and limitations
- Raising the Jewish-group threshold lowers wrong flags but raises missed hate against that group; the human review band is what keeps this acceptable.
- Adaptive attackers still evade the model most of the time; rate limits and human review are required.
- Group tags come from a lexicon of neutral terms and miss coded or implicit references.
- Several groups have fewer than 60 harmless test posts, so their rates carry wide intervals (table above).
- Retraining on user reports needs the label-drift check and trigger scan described in the report.
