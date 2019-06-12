import pandas as pd
import numpy as np
from config import Numerical_columns, Columns, Data_path, File_name, Nrows, Attack
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


def loadingKDD(path=Data_path+File_name, nrows=Nrows, attack_mode=True,
               numerical_columns=Numerical_columns, columns=Columns, attack=Attack,
               force_cat_col=None):

    df = pd.read_csv(path, names=columns, nrows=nrows)


    list_of_attacks = list(set(list(df.attack_type)))
    if attack_mode is None:
        if attack is None:
            pass
        else:
            df = df[(df.attack_type == attack) | (df.attack_type != "normal")]
    elif attack_mode:
        if attack is None:
            df = df[(df.attack_type != "normal")]
        else:
            df = df[(df.attack_type == attack)]
    else:
        df = df[df.attack_type == "normal"]
    df.attack_type = df.attack_type.apply(turn_attack)
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
        cat_col = df_one_hot_encoding.columns.to_list()
        for col in force_cat_col:
            if col not in cat_col:
                df_one_hot_encoding[col] = 0
        df_one_hot_encoding = df_one_hot_encoding[force_cat_col]
        cat_col = force_cat_col
    else:
        cat_col = df_one_hot_encoding.columns.to_list()

    scaler = MinMaxScaler()
    df_to_scale = pd.merge(df_numerical, df_one_hot_encoding, left_index=True, right_index=True)
    df_scaled = scaler.fit_transform(df_to_scale)

    X = df_scaled * 2 - 1
    Y = df.attack_type.values
    colnames = nc + cat_col
    return X, Y, cat_col, colnames
