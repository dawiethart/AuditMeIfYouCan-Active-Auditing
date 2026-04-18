# blackbox_api.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

class BlackBoxAPI:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(max_features=50000, stop_words="english"),
            LogisticRegression(solver="liblinear"),
        )

    def train(self, texts, labels, groups, biased=True):
        if biased:
            mask = (groups == "black") & (np.random.rand(len(groups)) < 0.9)
            labels = labels.astype(int).copy()
            labels[mask] = 1 - labels[mask]
        self.model.fit(texts, labels)

    def predict_scores(self, texts):
        return self.model.predict_proba(texts)[:, 1]
    
    def predict_scores_amazon(self, texts):
        return scores_amazon(texts)
    
    def predict_scores_openai(self, texts):
        return scores_openai(texts)
