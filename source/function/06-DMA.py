# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:09:27 2024

@author: Perat
"""
import importlib
import inspect

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import f_regression, r_regression
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVR

from source.common.const import DATASET, FILEPATH, STAGING_FILENAME
from source.common.helper import change_case

data_transformation = importlib.import_module("source.function.04-DT")

vars_6_data_mining_algorithms = {}
vars_6_1_rank_feature = {}
vars_6_1_model_exploration = {}

result = {}


def rank_feature(
    df: DataFrame,
    test_models: list[callable],
    score_funcs: list[callable],
    target: str,
) -> dict[str, callable]:
    global vars_6_1_rank_feature
    global result
    print("\t6.1.1 Objective Importance Feature")
    best_models = {}
    for test_model in test_models:
        result[test_model.__name__] = {}
        best_model_score = 0
        best_model = None
        for score_func in score_funcs:
            print("\t\tRank Feature With:", test_model.__name__, score_func.__name__)
            result[test_model.__name__][score_func.__name__], score = (
                data_transformation.data_reduction(
                    df=df,
                    test_model=test_model,
                    score_func=score_func,
                    target=target,
                    log=False,
                )
            )
            print("\t\t", test_model.__name__, score)
            if best_model is None or (score > best_model_score):
                best_model = result[test_model.__name__][score_func.__name__]
                best_model_score = score
        best_models[test_model.__name__] = best_model

    score_df = {}
    score_df_select = {}
    for score_func_name in result[list(result.keys())[0]].keys():
        score_df[score_func_name] = []
        score_df_select[score_func_name] = []
        for test_model_name, score_func_dict in result.items():
            feature_selection = score_func_dict[score_func_name]
            tmp_df_1 = pd.DataFrame(
                {
                    "field": [
                        feature_selection.feature_names_in_[i]
                        for i in range(len(feature_selection.scores_))
                    ],
                    f"{test_model_name}_score": [
                        feature_selection.scores_[i]
                        for i in range(len(feature_selection.scores_))
                    ],
                },
            ).set_index("field")
            tmp_df_1["abs"] = tmp_df_1[f"{test_model_name}_score"].abs()
            tmp_df_1.sort_values(by=["abs"], ascending=False, inplace=True)
            tmp_df_1.drop(["abs"], axis=1, inplace=True)
            score_df[score_func_name].append(tmp_df_1)
            tmp_df_2 = (
                pd.DataFrame(
                    {
                        "field": [
                            feature_selection.feature_names_in_[i]
                            for i in range(len(feature_selection.scores_))
                        ],
                        f"{test_model_name}_score": [
                            feature_selection.scores_[i]
                            for i in range(len(feature_selection.scores_))
                        ],
                    },
                )
                .set_index("field")
                .loc[feature_selection.get_feature_names_out(), :]
            )
            tmp_df_2["abs"] = tmp_df_2[f"{test_model_name}_score"].abs()
            tmp_df_2.sort_values(by=["abs"], ascending=False, inplace=True)
            tmp_df_2.drop(["abs"], axis=1, inplace=True)
            score_df_select[score_func_name].append(tmp_df_2)

    df_score = {}
    df_score_select = {}
    for score_func_name in score_df.keys():
        df_score[score_func_name] = pd.concat(
            score_df[score_func_name],
            axis=1,
            join="outer",
            copy=True,
        ).reset_index(drop=False)

        df_score[score_func_name].to_csv(
            f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DMA}_{score_func_name}_field_score.csv",
            index=False,
        )

        df_score_select[score_func_name] = pd.concat(
            score_df_select[score_func_name],
            axis=1,
            join="outer",
            copy=True,
        ).reset_index(drop=False)

        df_score_select[score_func_name].to_csv(
            f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DMA}_{score_func_name}_field_score_select.csv",
            index=False,
        )
    for name, model in best_models.items():
        print("\t\t", name, model.get_params())

    vars_6_1_rank_feature = inspect.currentframe().f_locals
    return best_models


def model_exploration(
    target: str,
    test_size: float,
    models_list: list[callable],
    hyper_params_list: dict[dict],
    df: DataFrame = None,
    selection_models: dict = None,
    in_x_train: DataFrame = None,
    in_x_test: DataFrame = None,
    in_y_train: DataFrame = None,
    in_y_test: DataFrame = None,
):
    global vars_6_1_model_exploration
    print("\t6.1.2 Objective Model Exploration")
    if selection_models is None:
        x = df.drop(target, axis=1)
        y = df[target]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )
    elif (
        selection_models is None
        and in_x_train is not None
        and in_x_test is not None
        and in_y_train is not None
        and in_y_test is not None
    ):
        x_train, x_test, y_train, y_test = in_x_train, in_x_test, in_y_train, in_y_test
    best_models = {}
    accuracy = ""
    for model in models_list:
        if (
            selection_models
            and in_x_train is not None
            and in_x_test is not None
            and in_y_train is not None
            and in_y_test is not None
        ):
            columns = selection_models[model.__name__].get_feature_names_out().tolist()
            x_train = in_x_train.drop(in_x_train.columns.difference(columns), axis=1)
            x_test = in_x_test.drop(in_x_test.columns.difference(columns), axis=1)
            y_train, y_test = in_y_train, in_y_test
            x, y = x_train, y_train
        elif (
            in_x_train is not None
            and in_x_test is not None
            and in_y_train is not None
            and in_y_test is not None
        ):
            x_train, x_test, y_train, y_test = (
                in_x_train,
                in_x_test,
                in_y_train,
                in_y_test,
            )
        elif selection_models:
            columns = selection_models[model.__name__].get_feature_names_out().tolist()
            x = df.drop(df.columns.difference(columns), axis=1)
            y = df[target]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=42
            )
        ini_model = model()
        tuning_model = GridSearchCV(
            ini_model,
            param_grid=hyper_params_list[model.__name__],
            scoring="neg_mean_squared_error",
            cv=3,
            verbose=1,
            error_score="raise",
        )
        tuning_model.fit(x, y)
        best_models[model.__name__] = {}
        best_models[model.__name__]["base"] = model
        best_models[model.__name__]["best_params_"] = tuning_model.best_params_
        best_models[model.__name__]["model"] = model(**tuning_model.best_params_)
        best_models[model.__name__]["model"].fit(x_train, y_train)
        model_pred = best_models[model.__name__]["model"].predict(x_test)
        best_models[model.__name__]["score"] = {}
        best_models[model.__name__]["score"]["MAE"] = mean_absolute_error(
            y_test, model_pred
        )
        best_models[model.__name__]["score"]["MSE"] = mean_squared_error(
            y_test, model_pred
        )
        best_models[model.__name__]["score"]["RMSE"] = np.sqrt(
            mean_squared_error(y_test, model_pred)
        )
        accuracy += (
            f"{model.__name__}\n"
            f"Hyperparams: {best_models[model.__name__]['best_params_']}\n"
            f"MAE: {best_models[model.__name__]['score']['MAE']}\n"
            f"MSE: {best_models[model.__name__]['score']['MSE']}\n"
            f"RMSE: {best_models[model.__name__]['score']['RMSE']}\n"
        )
    print(accuracy)
    vars_6_1_model_exploration = inspect.currentframe().f_locals
    return best_models


def data_mining_algorithm(df: DataFrame, target: str, function_kwargs: dict):
    global vars_6_data_mining_algorithms

    selection_models = rank_feature(
        df=df, target=target, **function_kwargs["rank_feature_kwargs"]
    )

    best_models = model_exploration(
        df=df,
        target=target,
        selection_models=selection_models,
        **function_kwargs["model_exploration"],
    )

    sorted_key = sorted(
        best_models,
        key=lambda x: (best_models[x]["score"]["MAE"], best_models[x]["score"]["MAE"]),
    )
    sorted_best_models = {}
    for key in sorted_key:
        sorted_best_models[key] = best_models[key]
    vars_6_data_mining_algorithms = inspect.currentframe().f_locals
    return selection_models, sorted_best_models


if __name__ == "__main__":
    df = pd.read_csv(f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}.csv")
    function_kwargs = {
        "rank_feature_kwargs": {
            "test_models": [LinearRegression, Ridge, SGDRegressor, LinearSVR],
            "score_funcs": [f_regression, r_regression],
        },
        "model_exploration": {
            "test_size": 0.2,
            "models_list": [LinearRegression, Ridge, SGDRegressor, LinearSVR],
            "hyper_params_list": {
                "LinearRegression": {},
                "Ridge": {
                    "alpha": [0.1, 0.5, 1, 2, 3],
                    "tol": [1e-3, 1e-4, 1e-5],
                    "solver": ["svd", "cholesky", "sparse_cg", "lsqr", "sag"],
                },
                "SGDRegressor": {
                    "loss": [
                        "squared_error",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ],
                    "penalty": ["l2", "l1", "elasticnet"],
                    "max_iter": [1000, 2000, 3000],
                    "tol": [1e-3, 1e-4],
                    "epsilon": [0.1, 0.01],
                    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                    "early_stopping": [True, False],
                },
                "LinearSVR": {
                    "epsilon": [0, 0.01, 0.001],
                    "tol": [1e-3, 1e-4, 1e-5],
                    "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                    "dual": ["auto"],
                },
            },
        },
    }
    data_mining_algorithm(
        df=df, target=change_case(DATASET.TARGET), function_kwargs=function_kwargs
    )
