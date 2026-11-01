"""
Toxicity classifier: TF-IDF (word 1-2 grams + character 2-5 grams) -> logistic regression.

Why this model: it trains in under a minute on a laptop, it is directly interpretable
(every feature has a weight), and the character n-grams make it partly resistant to
misspelling attacks, which we then test explicitly.
"""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .adversarial import normalize_text


class ModerationModel:
    def __init__(self, C=4.0, normalize=False, name="model"):
        self.C = C
        self.normalize = normalize
        self.name = name
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                    sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                    max_features=120000, sublinear_tf=True)
        self.clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

    def _prep(self, texts):
        texts = list(texts)
        return [normalize_text(t) for t in texts] if self.normalize else texts

    def _features(self, texts, fit=False):
        texts = self._prep(texts)
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, y, sample_weight=None):
        X = self._features(texts, fit=True)
        self.clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, texts):
        """Probability that each text is toxic."""
        if isinstance(texts, str):
            texts = [texts]
        return self.clf.predict_proba(self._features(texts))[:, 1]

    def predict_proba_2col(self, texts):
        """Two-column output, the format LIME expects."""
        p = self.predict_proba(texts)
        return np.column_stack([1 - p, p])

    def top_word_features(self, k=25):
        """Global explanation: word features with the largest positive / negative weights."""
        names = self.word.get_feature_names_out()
        w = self.clf.coef_[0][: len(names)]
        order = np.argsort(w)
        return ([(names[i], float(w[i])) for i in order[::-1][:k]],
                [(names[i], float(w[i])) for i in order[:k]])

    def word_weight(self, term):
        names = self.word.vocabulary_
        if term not in names:
            return 0.0
        return float(self.clf.coef_[0][names[term]])
