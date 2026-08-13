from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



def evaluate(
    y_true,
    y_pred
):


    result={}


    result["accuracy"] = accuracy_score(
        y_true,
        y_pred
    )


    result["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    result["recall"] = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    result["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    return resultevaluator.py
