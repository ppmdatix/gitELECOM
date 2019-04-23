from sklearn.ensemble import RandomForestClassifier

def loadIDS(mode="RandomForest"):
    modes = ["RandomForest"]
    assert mode in modes
    if mode == "RandomForest":
        clf = RandomForestClassifier()
    return clf


def trainIDS(clf, x_train, y_train):
    clf.fit(x_train, y_train)
    return clf

