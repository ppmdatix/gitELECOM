import pandas as pd
from config import Numerical_columns, Columns, Data_path, File_name, Nrows, Attack
from sklearn.preprocessing import MinMaxScaler


def turn_attack(x):
    if x == "normal":
        return 0.
    else:
        return 1.


def loadingKDD(path=Data_path+File_name, nrows=Nrows, attack_mode=True,
               numerical_columns=Numerical_columns, columns=Columns, attack=Attack):

    df = pd.read_csv(path, names=columns, nrows=nrows)

    if attack_mode is None:
        pass
    elif attack_mode:
        if attack is None:
            df = df[(df.attack_type != "normal")]
        else:
            df = df[(df.attack_type == attack)]
    else:
        df = df[df.attack_type == "normal"]
    df.attack_type = df.attack_type.apply(turn_attack)
    nc = numerical_columns + ["land", "logged_in",
                              "root_shell",
                              "is_host_login", "is_guest_login"]
    df_numerical = df[nc]
    df_numerical.reset_index(drop=True, inplace=True)

    categorical_columns = ["protocol_type","flag", "service", "su_attempted"]
    df_one_hot_encoding = df[categorical_columns]
    df_one_hot_encoding = pd.get_dummies(df_one_hot_encoding)
    df_one_hot_encoding.reset_index(drop=True, inplace=True)

    scaler = MinMaxScaler()
    df_to_scale = pd.merge(df_numerical, df_one_hot_encoding, left_index=True, right_index=True)
    df_scaled = scaler.fit_transform(df_to_scale)

    X = df_scaled * 2 - 1
    Y = df.attack_type.values
    columns = df_numerical.columns.to_list() + df_one_hot_encoding.columns.to_list()
    return X, Y, columns