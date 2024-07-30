# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:09:47 2024

@author: Perat
"""
import inspect

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from source.common.const import DATASET, FILEPATH, STAGING_FILENAME
from source.common.helper import change_case

vars_8_try_scaler = {}


def try_scaler(df: DataFrame, target: str, test_size: float):
    global vars_8_try_scaler
    x = df.drop(target, axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_new = scaler.transform(x)
    new_df = pd.DataFrame(x_new, columns=x.columns)
    new_df[target] = y
    x_train = pd.DataFrame(x_train, columns=x.columns)
    x_test = pd.DataFrame(x_test, columns=x.columns)
    vars_8_try_scaler = inspect.currentframe().f_locals
    return x_train, x_test, y_train, y_test, new_df


if __name__ == "__main__":
    df = pd.read_csv(f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}.csv")
    try_scaler(df=df, target=change_case(DATASET.TARGET), test_size=0.2)
