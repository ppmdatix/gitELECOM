import pandas as pd
from config import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def turn_attack(x):
    if x == "normal":
        return 0.
    else:
        return 1.


def loadingKDD(path=data_path+file_name, nrows=nrows, attack_mode=True):

    df = pd.read_csv(path, names=columns, nrows=nrows)
    if attack_mode:
        df = df[(df.attack_type == "normal") | (df.attack_type == attack)]
    else:
        df = df[df.attack_type == "normal"]
    df.attack_type = df.attack_type.apply(turn_attack)

    df_numerical = df[numerical_columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)
    df_scaled = pd.DataFrame(df_scaled, columns=numerical_columns)

    X = df_scaled.values
    Y = df.attack_type.values
    return X, Y