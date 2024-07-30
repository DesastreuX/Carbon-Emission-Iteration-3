# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:09:39 2024

@author: Perat
"""

# 07 - Data Mining

import importlib
import inspect

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_selection import f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from source.common.const import DATASET, FILEPATH, STAGING_FILENAME
from source.common.helper import change_case

data_transformation = importlib.import_module("source.function.04-DT")

vars_7_data_mining = {}
vars_7_1_creating_test = {}


def creating_test(
    df: DataFrame,
    test_size: float,
    target: str,
    feature_selection_model: object = None,
) -> tuple[DataFrame]:
    global vars_7_1_creating_test
    if feature_selection_model is not None:
        columns = feature_selection_model.get_feature_names_out().tolist()
    else:
        columns = [target]
    x = df.drop(df.columns.difference(columns), axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )
    vars_7_1_creating_test = inspect.currentframe().f_locals
    return x_train, x_test, y_train, y_test


def do_data_mining(x_input: DataFrame, prediction_model: object) -> Series:
    return prediction_model.predict(x_input)


def test_model_result(y_test: Series, y_pred: Series) -> dict:
    result = {}
    result["MAE"] = mean_absolute_error(y_test, y_pred)
    result["MSE"] = mean_squared_error(y_test, y_pred)
    result["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
    result["R2 Score"] = r2_score(y_true=y_test, y_pred=y_pred)
    return result


def data_mining(
    target: str,
    option: str,
    function_kwargs: dict,
):
    global vars_7_data_mining
    if option.lower() == "creating_test":
        print("\t7.1 Creating Test")
        x_train, x_test, y_train, y_test = creating_test(
            target=target, **function_kwargs["creating_test_kwargs"]
        )
        vars_7_data_mining = inspect.currentframe().f_locals
        return x_train, x_test, y_train, y_test
    elif option.lower() == "test_model":
        print("\t7.2 Testing Model")
        y_pred = do_data_mining(
            **function_kwargs["test_model_kwargs"]["do_data_mining_kwargs"]
        )
        score = test_model_result(
            y_pred=y_pred,
            **function_kwargs["test_model_kwargs"]["test_model_result_kwargs"],
        )
        print(f"Model MAE: {score['MAE']}.")
        print(f"Model MSE: {score['MSE']}.")
        print(f"Model RMSE: {score['RMSE']}.")
        print(f"Model R2 Score: {score['R2 Score']}.")
        vars_7_data_mining = inspect.currentframe().f_locals
        return score
    elif option.lower() == "model_predict":
        vars_7_data_mining = inspect.currentframe().f_locals
        return do_data_mining(**function_kwargs["model_predict"])
    else:
        raise


if __name__ == "__main__":
    df = pd.read_csv(f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}.csv")
    feature_selection_model, _ = data_transformation.data_reduction(
        df=df,
        test_model=SGDRegressor,
        score_func=f_regression,
        target=change_case(DATASET.TARGET),
        print_score=True,
    )
    function_kwargs = {
        "creating_test_kwargs": {
            "df": df,
            "test_size": 0.2,
            "feature_selection_model": feature_selection_model,
        },
    }
    x_train, x_test, y_train, y_test = data_mining(
        target=change_case(DATASET.TARGET),
        option="creating_test",
        function_kwargs=function_kwargs,
    )
    hyper = {
        "alpha": 0.0001,
        "average": False,
        "early_stopping": True,
        "epsilon": 0.01,
        "eta0": 0.01,
        "fit_intercept": True,
        "l1_ratio": 0.15,
        "learning_rate": "adaptive",
        "loss": "squared_error",
        "max_iter": 10000,
        "n_iter_no_change": 5,
        "penalty": "l2",
        "power_t": 0.25,
        "random_state": None,
        "shuffle": True,
        "tol": 0.001,
        "validation_fraction": 0.1,
        "verbose": 0,
        "warm_start": False,
    }
    model = SGDRegressor(**hyper)
    model.fit(x_train, y_train)
    function_kwargs["test_model_kwargs"] = {
        "do_data_mining_kwargs": {
            "x_input": x_test,
            "prediction_model": model,
        },
        "test_model_result_kwargs": {"y_test": y_test},
    }
    data_mining(
        target=change_case(DATASET.TARGET),
        option="test_model",
        function_kwargs=function_kwargs,
    )
