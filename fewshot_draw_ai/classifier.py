import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



class FewShotClassifier:


    def __init__(self):

        self.scaler = StandardScaler()


        self.model = LogisticRegression(
            C=0.5,
            max_iter=2000,
            random_state=2026
        )



    def fit(
        self,
        features,
        labels
    ):


        features = self.scaler.fit_transform(
            features
        )


        self.model.fit(
            features,
            labels
        )



    def predict(
        self,
        features
    ):


        features = self.scaler.transform(
            features
        )


        return self.model.predict(
            features
        )



    def predict_proba(
        self,
        features
    ):


        features=self.scaler.transform(
            features
        )


        return self.model.predict_proba(
            features
        )
