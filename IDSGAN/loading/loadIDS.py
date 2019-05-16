from sklearn.ensemble import RandomForestClassifier


def loadIDS(mode="RandomForest",
            n_estimators=10,
            max_depth=None,
            n_jobs=-1):
    modes = ["RandomForest"]
    assert mode in modes
    if mode == "RandomForest":
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,n_jobs=n_jobs)
    return clf


def trainIDS(clf, x_train, y_train):
    clf.fit(x_train, y_train)
    return clf

