import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def loadData(path_data="/home/peseux/Downloads/UNSW-NB15/UNSW-NB15_1.csv",
             path_features="/home/peseux/Downloads/UNSW-NB15/NUSW-NB15_features.csv",
             nrows=100,
             test_size=0.2,
             attacks=True):

    def turnNanOk(x):
        if x!= x:
            return "Ok"
        return x

    def turnSwin(x):
        if x:
            return 1
        return 0

    # Loading features
    df_features = pd.read_csv(path_features)
    names = list(df_features.Name)
    df_features_float = df_features[df_features["Type "] == "Float"]
    name_float = list(df_features_float["Name"])
    int_cat_to_use = ["Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
                      "ct_ftp_cmd", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm",
                      "ct_dst_sport_ltm", "ct_dst_src_ltm"]

    # Loading data
    df = pd.read_csv(path_data, names=names, nrows=nrows)
    print(df.shape)

    # Prepossessing on categorical data
    df.swin = df.swin.apply(turnSwin)
    df.dwin = df.dwin.apply(turnSwin)
    df.attack_cat = df.attack_cat.apply(turnNanOk)
    df.dropna(inplace=True)
    df.Label = df.Label.apply(int)
    # attacks = list(df.attack_cat.unique())

    # Keeping DOS attacks and benign ones
    if attacks:
        df_dos = df[(df.attack_cat == "Ok") | (df.attack_cat == "DoS")]
    else:
        df_dos = df[df.attack_cat == "DoS"]

    df_dos_float = df_dos[name_float + int_cat_to_use]

    # Scaling
    scaler = StandardScaler()
    final_data = scaler.fit_transform(df_dos_float)
    final_data = pd.DataFrame(final_data, columns=name_float + int_cat_to_use)

    # tanh
    final_data = final_data.apply(np.tanh)

    Y = df_dos[["Label"]]
    final_data["Label"] = list(Y.Label)
    train, test = train_test_split(final_data, test_size=test_size)
    y_tr = train[["Label"]]
    x_tr = train.drop("Label", axis=1)
    y_te = test[["Label"]]
    x_te = test.drop("Label", axis=1)

    return x_tr.values, y_tr.values, x_te.values, y_te.values





