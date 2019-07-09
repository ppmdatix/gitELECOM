import pandas as pd
import numpy as np
from config import Numerical_columns, Columns, File_name, Nrows, Attack, Data_path_home, Data_path_work
from sklearn.preprocessing import MinMaxScaler


def turn_attack(x):
    if x == "normal":
        return 0.
    else:
        return 1.


def turn_attack_label(x, liste):
    return liste.index(liste)


def logplus1(x):
    return np.log(x+1)


def loadingKDD(path=None, nrows=Nrows, attack_mode=True,
               numerical_columns=Numerical_columns, columns=Columns, attack=Attack,
               force_cat_col=None,
               place="home",
               data_path_home=Data_path_home,
               data_path_work=Data_path_work,
               log_transform=True):
    assert (place is None) or place in ["work", "home"], "place argument should be None work or home"
    if place == "home":
        path = data_path_home + File_name
    elif place == "work":
        path = data_path_work + File_name

    df = pd.read_csv(path, names=columns, nrows=nrows)

    list_of_attacks = list(set(list(df.attack_type)))
    if attack_mode is None:
        if attack is None:
            pass
        else:
            df = df[(df.attack_type == attack) | (df.attack_type == "normal")]
    elif attack_mode:
        if attack is None:
            df = df[(df.attack_type != "normal")]
        else:
            df = df[(df.attack_type == attack)]
    else:
        df = df[df.attack_type == "normal"]
    df.attack_type = df.attack_type.apply(turn_attack)
    if log_transform:
        for col in numerical_columns:
            df[col] = df[col].apply(logplus1)

    nc = numerical_columns + ["land", "logged_in",
                              "root_shell",
                              "is_host_login", "is_guest_login", "su_attempted"]
    df_numerical = df[nc]
    df_numerical.reset_index(drop=True, inplace=True)

    categorical_columns = ["protocol_type", "flag", "service"]
    df_one_hot_encoding = df[categorical_columns]
    df_one_hot_encoding = pd.get_dummies(df_one_hot_encoding)
    df_one_hot_encoding.reset_index(drop=True, inplace=True)
    if force_cat_col is not None:
        try:
            cat_col = df_one_hot_encoding.columns.to_list()
        except:
            cat_col = df_one_hot_encoding.columns.tolist()
        for col in force_cat_col:
            if col not in cat_col:
                df_one_hot_encoding[col] = 0
        df_one_hot_encoding = df_one_hot_encoding[force_cat_col]
        cat_col = force_cat_col
    else:
        try:
            cat_col = df_one_hot_encoding.columns.to_list()
        except:
            cat_col = df_one_hot_encoding.columns.tolist()
    scaler = MinMaxScaler()
    df_to_scale = pd.merge(df_numerical, df_one_hot_encoding, left_index=True, right_index=True)
    df_scaled = scaler.fit_transform(df_to_scale)

    x_data = df_scaled * 2 - 1
    y_data = df.attack_type.values
    colnames = nc + cat_col
    return x_data, y_data, cat_col, colnames
