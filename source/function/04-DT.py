# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:09:03 2024

@author: Perat
"""

# 04 - Data Transformation

import inspect

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline

from source.common.const import DATASET, FILEPATH, STAGING_FILENAME
from source.common.helper import change_case, dataframe_detail_describe

vars_4_data_transformation = {}
vars_4_1_data_reduction = {}


# 04 - 4.1 Data Reduction
def data_reduction(
    df: DataFrame,
    test_model: callable,
    score_func: callable,
    target: str,
    log: bool = True,
    print_score: bool = False,
) -> SelectKBest:
    global vars_4_1_data_reduction
    if log:
        print("\t4.1 Data Reduction")
    x = df.drop(target, axis=1)
    y = df[target]

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    if test_model.__name__ == "LinearSVR":
        model = test_model(dual="auto")
    else:
        model = test_model()
    feature_selection = SelectKBest(score_func=score_func)
    pipeline = Pipeline(steps=[("sel", feature_selection), ("lr", model)])
    grid = dict()
    grid["sel__k"] = [i for i in range(1, x.shape[1] + 1)]
    search = GridSearchCV(pipeline, grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=cv)
    results = search.fit(x, y)
    feature_selection = SelectKBest(score_func=score_func, k=results.best_params_["sel__k"])
    feature_selection.fit(x, y)
    if print_score:
        print("Best Config: k = %s" % results.best_params_["sel__k"])
        score_dict = {
            feature_selection.feature_names_in_[i]: feature_selection.scores_[i]
            for i in range(len(feature_selection.scores_))
        }
        score_dict = sorted(score_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        temp_df = pd.DataFrame.from_dict(score_dict)
        temp_df.plot.bar(0, 1)
        for feature, score in score_dict:
            print(f"{feature}: {score}")
    vars_4_1_data_reduction = inspect.currentframe().f_locals
    return feature_selection, search.best_score_


# 04 - 4.2 Data Projection
def data_projection(df: DataFrame):
    sns.histplot(data=df, x="vehicle_monthly_distance_km").set(title="Original")
    temp_df = df.copy()
    temp_df["vehicle_monthly_distance_km_sqrt"] = np.sqrt(np.abs(temp_df["vehicle_monthly_distance_km"]))
    temp_df["vehicle_monthly_distance_km_log"] = np.log(np.abs(temp_df["vehicle_monthly_distance_km"]))
    temp_df["vehicle_monthly_distance_km_log10"] = np.log10(np.abs(temp_df["vehicle_monthly_distance_km"]))
    sns.histplot(data=temp_df, x="vehicle_monthly_distance_km_sqrt").set(title="Square Root")
    sns.histplot(data=temp_df, x="vehicle_monthly_distance_km_log").set(title="LogN")
    sns.histplot(data=temp_df, x="vehicle_monthly_distance_km_log10").set(title="Log10")


def data_transformation(df: DataFrame, target: str, function_kwargs: dict) -> DataFrame:
    global vars_4_data_transformation
    feature_selection, _ = data_reduction(df=df, target=target, **function_kwargs["data_reduction_kwargs"])
    columns = feature_selection.get_feature_names_out().tolist()
    columns.append(target)
    df.drop(df.columns.difference(columns), axis=1, inplace=True)
    df.to_csv(
        f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DT}.csv",
        index=False,
    )

    data_projection(df=df)

    info_table = dataframe_detail_describe(df=df, target_name=target)

    vars_4_data_transformation = inspect.currentframe().f_locals


if __name__ == "__main__":
    df = pd.read_csv(f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}.csv")

    function_kwargs = {
        "data_reduction_kwargs": {
            "test_model": LinearRegression,
            "score_func": f_regression,
            "print_score": True,
        }
    }

    from sklearn.linear_model import SGDRegressor

    function_kwargs = {
        "data_reduction_kwargs": {
            "test_model": SGDRegressor,
            "score_func": f_regression,
            "print_score": True,
        }
    }
    data_transformation(df=df, target=change_case(DATASET.TARGET), function_kwargs=function_kwargs)
