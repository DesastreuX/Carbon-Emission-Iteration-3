# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:08:28 2024

@author: Perat
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.common.const import DATASET, FILEPATH
from source.common.helper import dataframe_detail_describe, rename_columns

# 02 - Data Understanding.

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

filename = FILEPATH.RAW

df = pd.read_csv(filename)

# 2.2 Describing Data
df.info()
info_table = dataframe_detail_describe(df=df, target_name=DATASET.TARGET)


# 2.3 Data Exploration
rename_columns(df=df)

LE = LabelEncoder()
for column_name in df.columns:
    if not is_numeric_dtype(df[column_name]) and column_name != DATASET.TARGET:
        df[column_name] = LE.fit_transform(df[column_name])


x = df.drop(["carbon_emission"], axis=1)
y = df["carbon_emission"]

# Predictor Importance
models = {
    "cart": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "xgboost": XGBRegressor(),
}
importance_df = pd.DataFrame()
for model_name, model_obj in models.items():
    model_obj.fit(x, y)
    # get importance
    importance = model_obj.feature_importances_
    # summarize feature importance
    importance_dict = {}
    for i, v in enumerate(importance):
        importance_dict[df.columns[i]] = v
    if importance_df.empty:
        importance_df = pd.DataFrame(
            importance_dict.items(),
            columns=[
                "feature",
                f"{model_name}_importance",
            ],
        ).set_index("feature")
    else:
        importance_df = pd.concat(
            [
                importance_df,
                pd.DataFrame(
                    importance_dict.items(),
                    columns=[
                        "feature",
                        f"{model_name}_importance",
                    ],
                ).set_index("feature"),
            ],
            axis=1,
            join="inner",
            copy=False,
        )
importance_df = importance_df.assign(avg=importance_df.mean(axis=1))
importance_df.sort_values(by=["avg"], inplace=True, ascending=False)
