from evaluator import evaluate


test_features=[]


for x in test_coords:

    f = feature_model.encode(
        x
    )

    test_features.append(f)



prediction = classifier.predict(
    test_features
)



score=evaluate(
    test_labels,
    prediction
)


print(score)
