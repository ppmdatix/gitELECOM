from sklearn.metrics import confusion_matrix as confusion_matrix, precision_score,f1_score, recall_score


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    ps = precision_score(y_true=y_true, y_pred=y_pred)
    rs = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    return {"confusion_matrix": cm, "precision": ps, "recall": rs, "f1_score": f1}