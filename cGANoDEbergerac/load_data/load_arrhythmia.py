import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def load_arrhythmia(test_size=.33):
    data = scipy.io.loadmat("../arrhythmia/arrhythmia.mat")
    x = data["X"]
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)*2 -1
    y = data["y"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_train_cv, y_train, y_train_cv = train_test_split(x_train, y_train, test_size=test_size)
    return x_train, x_test, y_train, y_test, x_train_cv, y_train_cv
