# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:28:04 2024

@author: Perat
"""
from sklearn.feature_selection import f_regression, r_regression
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.svm import LinearSVR

from source.common.const import FILEPATH

data_preparation_kwargs = {
    "data_integration_kwargs": {
        "filepath_list": [
            FILEPATH.CARBON_EMISSION_AMOUNT,
            FILEPATH.CARBON_EMISSION_HEALTH,
            FILEPATH.CARBON_EMISSION_LIFESTYLE,
            FILEPATH.CARBON_EMISSION_TRAVEL,
            FILEPATH.CARBON_EMISSION_WASTE,
        ],
        "file_type": "csv",
        "join_key": "ID",
        "drop_key": True,
    },
    "data_selection_kwargs": {
        "exclude_columns": [
            "how_long_t_v_p_c_daily_hour",
            "how_often_shower",
            "energy_efficiency",
        ],
    },
    "data_cleaning_kwargs": {
        "do_not_drop": ["vehicle_type"],
        "column_type_mapping": {"recycling": set, "cooking_with": set},
    },
    "data_construction_kwargs": {
        "explode_and_count_columns": ["recycling", "cooking_with"],
        "quantile_columns": {"monthly_grocery_bill": 3},
        "standardize_columns": [
            "monthly_grocery_bill",
            "vehicle_monthly_distance_km",
            "waste_bag_weekly_count",
            "how_many_new_clothes_monthly",
            "how_long_internet_daily_hour",
        ],
    },
}

data_transformation_kwargs = {
    "data_reduction_kwargs": {
        "test_model": LinearRegression,
        "score_func": f_regression,
        "log": True,
    }
}

data_mining_algorithm_kwargs = {
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
                "max_iter": [10000],
                "tol": [1e-3, 1e-4],
                "epsilon": [0.1, 0.01],
                "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                "early_stopping": [True],
            },
            "LinearSVR": {
                "epsilon": [0, 0.01, 0.001],
                "tol": [1e-3, 1e-4, 1e-5],
                "max_iter": [1000],
                "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                "dual": ["auto"],
            },
        },
    },
}

data_mining_kwargs = {
    "creating_test_kwargs": {
        "test_size": 0.2,
    },
    "test_model_kwargs": {
        "do_data_mining_kwargs": {},
        "test_model_result_kwargs": {},
    },
}
