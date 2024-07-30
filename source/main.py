# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:16:11 2024

@author: Perat
"""
import importlib
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (LearningCurveDisplay, ShuffleSplit,
                                     train_test_split)

from source.common.const import DATASET
from source.common.helper import change_case
from source.common.keyword_args import (data_mining_algorithm_kwargs,
                                        data_mining_kwargs,
                                        data_preparation_kwargs,
                                        data_transformation_kwargs)

data_preparation = importlib.import_module("source.function.03-DP")
data_transformation = importlib.import_module("source.function.04-DT")
data_mining_algorithm = importlib.import_module("source.function.06-DMA")
data_mining = importlib.import_module("source.function.07-DM")
iteration = importlib.import_module("source.function.08-IN")

# 03 - Data Preparation
print("3 Data Preparation")
df = data_preparation.data_preparation(
    function_kwargs=data_preparation_kwargs,
    target=change_case(DATASET.TARGET),
)

df_03 = df.copy()

# 04 - Data Transformation
print("4 Data Transformation")
data_transformation.data_transformation(
    df=df,
    target=change_case(DATASET.TARGET),
    function_kwargs=data_transformation_kwargs,
)

df_04 = df.copy()

# 06 - Data Mining Algorithm
print("6 Data Mining Algorithm")
selection_models, sorted_best_models = data_mining_algorithm.data_mining_algorithm(
    df=df_03,
    target=change_case(DATASET.TARGET),
    function_kwargs=data_mining_algorithm_kwargs,
)

# 07 - Data Mining
print("7 Data Mining")
data_mining_kwargs["creating_test_kwargs"]["df"] = df_03
data_mining_kwargs["creating_test_kwargs"]["feature_selection_model"] = (
    selection_models[next(iter(sorted_best_models))]
)
x_train, x_test, y_train, y_test = data_mining.data_mining(
    target=change_case(DATASET.TARGET),
    option="creating_test",
    function_kwargs=data_mining_kwargs,
)
data_mining_kwargs["test_model_kwargs"]["do_data_mining_kwargs"]["x_input"] = x_test
data_mining_kwargs["test_model_kwargs"]["do_data_mining_kwargs"]["prediction_model"] = (
    sorted_best_models[next(iter(sorted_best_models))]["model"]
)
data_mining_kwargs["test_model_kwargs"]["test_model_result_kwargs"]["y_test"] = y_test
data_mining.data_mining(
    target=change_case(DATASET.TARGET),
    option="test_model",
    function_kwargs=data_mining_kwargs,
)

# plotting learning curve

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
columns = (
    selection_models[next(iter(sorted_best_models))].get_feature_names_out().tolist()
)
x = df.drop(df.columns.difference(columns), axis=1)
y = df[change_case(DATASET.TARGET)]
common_params = {
    "X": x,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}
plot_model = sorted_best_models[next(iter(sorted_best_models))]["base"](
    **sorted_best_models[next(iter(sorted_best_models))]["best_params_"]
)

LearningCurveDisplay.from_estimator(plot_model, **common_params, ax=ax)
handles, label = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["Training Score", "Test Score"])
ax.set_title(f"Learning Curve for {plot_model.__class__.__name__}")

# 08 Iteration
print("08 Iteration")
# Testing StandardScaler
x_train, x_test, y_train, y_test, df_08 = iteration.try_scaler(
    df=df_03, target=change_case(DATASET.TARGET), test_size=0.2
)


data_mining_algorithm_08_kwargs = deepcopy(data_mining_algorithm_kwargs)
data_mining_algorithm_08_kwargs["model_exploration"]["in_x_train"] = x_train
data_mining_algorithm_08_kwargs["model_exploration"]["in_x_test"] = x_test
data_mining_algorithm_08_kwargs["model_exploration"]["in_y_train"] = y_train
data_mining_algorithm_08_kwargs["model_exploration"]["in_y_test"] = y_test
data_mining_algorithm_08_kwargs["model_exploration"]["hyper_params_list"][
    "SGDRegressor"
]["max_iter"] = [30000]
selection_models = data_mining_algorithm.rank_feature(
    df=df,
    target=change_case(DATASET.TARGET),
    **data_mining_algorithm_08_kwargs["rank_feature_kwargs"],
)

best_models = data_mining_algorithm.model_exploration(
    target=change_case(DATASET.TARGET),
    selection_models=selection_models,
    **data_mining_algorithm_08_kwargs["model_exploration"],
)

sorted_key = sorted(
    best_models,
    key=lambda x: (best_models[x]["score"]["MAE"], best_models[x]["score"]["MAE"]),
)
sorted_best_models = {}
for key in sorted_key:
    sorted_best_models[key] = best_models[key]
data_mining_kwargs_08 = deepcopy(data_mining_kwargs)
data_mining_kwargs_08["creating_test_kwargs"]["df"] = df_08
data_mining_kwargs_08["creating_test_kwargs"]["feature_selection_model"] = (
    selection_models[next(iter(sorted_best_models))]
)
x_train, x_test, y_train, y_test = data_mining.data_mining(
    target=change_case(DATASET.TARGET),
    option="creating_test",
    function_kwargs=data_mining_kwargs_08,
)
data_mining_kwargs_08["test_model_kwargs"]["do_data_mining_kwargs"]["x_input"] = x_test
data_mining_kwargs_08["test_model_kwargs"]["do_data_mining_kwargs"][
    "prediction_model"
] = sorted_best_models[next(iter(sorted_best_models))]["model"]
data_mining_kwargs_08["test_model_kwargs"]["test_model_result_kwargs"][
    "y_test"
] = y_test
data_mining.data_mining(
    target=change_case(DATASET.TARGET),
    option="test_model",
    function_kwargs=data_mining_kwargs_08,
)

# Testing Using All Features
data_mining_algorithm_08_kwargs = deepcopy(data_mining_algorithm_kwargs)
data_mining_algorithm_08_kwargs["model_exploration"]["df"] = df_03
best_models = data_mining_algorithm.model_exploration(
    target=change_case(DATASET.TARGET),
    **data_mining_algorithm_08_kwargs["model_exploration"],
)
sorted_key = sorted(
    best_models,
    key=lambda x: (best_models[x]["score"]["MAE"], best_models[x]["score"]["MAE"]),
)
sorted_best_models = {}
for key in sorted_key:
    sorted_best_models[key] = best_models[key]

x = df_03.drop([change_case(DATASET.TARGET)], axis=1)
y = df_03[change_case(DATASET.TARGET)]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
data_mining_kwargs_08 = deepcopy(data_mining_kwargs)
data_mining_kwargs_08["creating_test_kwargs"]["df"] = df_03
data_mining_kwargs_08["creating_test_kwargs"]["feature_selection_model"] = None
data_mining_kwargs_08["test_model_kwargs"]["do_data_mining_kwargs"]["x_input"] = x_test
data_mining_kwargs_08["test_model_kwargs"]["do_data_mining_kwargs"][
    "prediction_model"
] = sorted_best_models[next(iter(sorted_best_models))]["model"]
data_mining_kwargs_08["test_model_kwargs"]["test_model_result_kwargs"][
    "y_test"
] = y_test
data_mining.data_mining(
    target=change_case(DATASET.TARGET),
    option="test_model",
    function_kwargs=data_mining_kwargs_08,
)
