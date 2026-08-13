from model import FewShotAdaptiveModel
from classifier import FewShotClassifier


# train data

train_features=[]


feature_model = FewShotAdaptiveModel()



for x in train_coords:

    f = feature_model.encode(
        x
    )

    train_features.append(f)



classifier = FewShotClassifier()


classifier.fit(
    train_features,
    train_labels
)
