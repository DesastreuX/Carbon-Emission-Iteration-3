# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:08:42 2024

@author: Perat
"""

# 03 - Data Preparation

import inspect
from ast import literal_eval

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype, is_string_dtype
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from source.common.const import DATASET, FILEPATH, STAGING_FILENAME
from source.common.exception import (DataIntegrationError, InvalidInput,
                                     UnsupportedFileType)
from source.common.helper import (change_case, dataframe_detail_describe,
                                  rename_columns)

vars_3_data_preparation = {}
vars_3_1_data_selection = {}
vars_3_2_data_cleansing = {}
vars_3_3_data_construction = {}
vars_3_4_data_integration = {}


def drop_outlier_Z_score_threshold(df: DataFrame, z_threshold: int, target_column: str):
    is_extreme_exist = True
    while is_extreme_exist:
        outlier_indices = set()
        for column_name in df.columns:
            if is_numeric_dtype(df[column_name]) and column_name != change_case(
                target_column
            ):
                z = np.abs(stats.zscore(df[column_name]))
                o = np.where(z > z_threshold)[0]
                outlier_indices |= set(o)
        if outlier_indices:
            df.drop(outlier_indices, inplace=True, axis=0)
            df.reset_index(drop=True, inplace=True)
        else:
            is_extreme_exist = False


def convert_string_represented_list_like_to_list_like(
    column_type_mapping: dict[str, callable],
    df: DataFrame,
):
    for column_name, column_type in column_type_mapping.items():
        df.loc[:, column_name] = df[column_name].apply(literal_eval).apply(column_type)


def drop_inconsistency(df: DataFrame):
    inconsistence_rows = set()
    for column_name in df.columns:
        if is_string_dtype(df[column_name]):
            distribution_df = DataFrame(
                df[column_name].value_counts(), columns=["value", "count"]
            )
            distribution_df["inconsistence"] = distribution_df["count"].apply(
                lambda x: (x / distribution_df["count"].sum())
            )
            distribution_df["is_inconsistence"] = distribution_df["count"].apply(
                lambda x: (x / distribution_df["count"].sum()) < 0.005
            )
            inconsistence_rows |= set(
                df.index[
                    df[column_name].isin(
                        distribution_df[
                            distribution_df["is_inconsistence"]
                        ].index.values
                    )
                ]
            )
    df.drop(
        inconsistence_rows,
        inplace=True,
        axis=0,
    )
    df.reset_index(drop=True, inplace=True)


def explode_list_columns(column_name: str, df: DataFrame):
    value_set = set.union(*df[column_name].tolist())
    for value in sorted(value_set):
        new_column_name = f"{column_name}_{value.lower()}"
        df.loc[:, new_column_name] = df[column_name].apply(lambda x: value in x)


# 03 - 3.1 Data Selection
def data_selection(df: DataFrame, exclude_columns: list[str]):
    global vars_3_1_data_selection

    not_exist_columns = set(exclude_columns) - set(df.columns)
    if not_exist_columns:
        raise InvalidInput(f"Error: Exclude columns not existed: {not_exist_columns}")
    else:
        df.drop(
            exclude_columns,
            inplace=True,
            axis=1,
        )

    vars_3_1_data_selection = inspect.currentframe().f_locals


# 03 - 3.2 Data Cleaning
def data_cleaning(
    df: DataFrame,
    target: str,
    do_not_drop: list[str],
    column_type_mapping: dict[str, callable],
    log: bool = True,
):
    global vars_3_2_data_cleansing

    if log:
        print("\t\tOriginal DataFrame Shape:", df.shape)
    other_cols = df.columns.difference(do_not_drop)
    df.drop(
        df.index[df[other_cols].isna().any(axis=1)],
        inplace=True,
        axis=0,
    )
    df.reset_index(drop=True, inplace=True)
    if log:
        print("\t\tDataFrame Shape after Removing NaN:", df.shape)

    df.fillna(value={k: "not applicable" for k in do_not_drop}, inplace=True)

    info_table_3_2_1 = dataframe_detail_describe(df=df, target_name=target)

    if log:
        print("\t\tConvert String Represented List to List Like.")
    convert_string_represented_list_like_to_list_like(
        column_type_mapping=column_type_mapping, df=df
    )

    if log:
        print("\t\tOriginal DataFrame Shape:", df.shape)
    drop_inconsistency(df=df)
    if log:
        print("\t\tDataFrame Shape after Removing Inconsistency:", df.shape)

    info_table_3_2_2 = dataframe_detail_describe(df=df, target_name=target)

    if log:
        print("\t\tOriginal DataFrame Shape:", df.shape)
    drop_outlier_Z_score_threshold(
        df=df,
        z_threshold=3,
        target_column=target,
    )
    if log:
        print("\t\tDataFrame Shape after Removing Outliers:", df.shape)

    info_table_3_2_3 = dataframe_detail_describe(df=df, target_name=target)

    vars_3_2_data_cleansing = inspect.currentframe().f_locals


# 03 - 3.3 Data Construction
def data_construction_carbon_emission(
    explode_and_count_columns: list[str],
    quantile_columns: dict[str, int],
    standardize_columns: list[str],
    target: str,
    df: DataFrame,
):
    global vars_3_3_data_construction

    for explode_column in explode_and_count_columns:
        explode_list_columns(column_name=explode_column, df=df)
        count_column_name = f"{explode_column}_count"
        df.loc[:, count_column_name] = df[explode_column].apply(len)

    for column_name, quantile_number in quantile_columns.items():
        quantile_bin = [
            df[column_name].quantile(i / quantile_number)
            for i in range(quantile_number + 1)
        ]
        quantile_column_name = f"{column_name}_quantile"
        df.loc[:, quantile_column_name] = pd.cut(
            df[column_name],
            bins=quantile_bin,
            labels=[i + 1 for i in range(quantile_number)],
            include_lowest=True,
        ).cat.codes

    for standardize_column in standardize_columns:
        df[standardize_column] = stats.zscore(df[standardize_column], nan_policy="omit")

    LE = LabelEncoder()
    for column_name in df.columns:
        if column_name != target:
            if is_string_dtype(df[column_name]):
                df.loc[:, column_name] = LE.fit_transform(df[column_name])
            elif any(
                isinstance(obj, tuple([list, dict, set, tuple]))
                for obj in df[column_name]
            ):
                df.loc[:, column_name] = LE.fit_transform(df[column_name].astype(str))

    info_table_3_3 = dataframe_detail_describe(df=df, target_name=target)

    vars_3_3_data_construction = inspect.currentframe().f_locals


# 03 - 3.4 Data Integration
def data_integration(
    filepath_list: list[str],
    file_type: str,
    join_key: str,
    target: str,
    drop_key: bool,
    *args,
    **kwargs: str,
) -> DataFrame:
    global vars_3_4_data_integration
    try:
        if file_type.lower() == "csv":
            read_function = pd.read_csv
        else:
            raise UnsupportedFileType("Error: Input File Type not Support.")

        list_df = []
        for i, filepath in enumerate(filepath_list):
            list_df.append(read_function(filepath, *args, **kwargs).set_index(join_key))

        df = pd.concat(
            list_df,
            axis=1,
            join="inner",
            copy=True,
        ).reset_index(drop=drop_key)

        info_table_3_4 = dataframe_detail_describe(df=df, target_name=target)

        vars_3_4_data_integration = inspect.currentframe().f_locals

        return df
    except Exception as e:
        error_message = str(e)
        raise DataIntegrationError(error_message)


def data_preparation(function_kwargs: dict[str, dict], target: str, log: bool = True):
    global vars_3_data_preparation

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # filename = FILEPATH.RAW
    # df = pd.read_csv(filename)
    if log:
        print("\t3.4 Data Integration")
    df = data_integration(target=target, **function_kwargs["data_integration_kwargs"])

    rename_columns(df=df)

    if log:
        print("\t3.1 Data Selection")
    data_selection(df=df, **function_kwargs["data_selection_kwargs"])

    df.to_csv(
        f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}-1-selection.csv",
        index=False,
    )
    if log:
        print("\t3.2 Data Cleaning")
    data_cleaning(df=df, target=target, **function_kwargs["data_cleaning_kwargs"])

    df.to_csv(
        f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}-2-cleansing.csv",
        index=False,
    )
    if log:
        print("\t3.3 Data Construction")
    data_construction_carbon_emission(
        df=df, target=target, **function_kwargs["data_construction_kwargs"]
    )

    df.to_csv(
        f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}-3-construction.csv",
        index=False,
    )

    df.to_csv(
        f"{FILEPATH.TEMP_STAGING_PATH}/{STAGING_FILENAME.DP}.csv",
        index=False,
    )
    info_table = dataframe_detail_describe(df=df, target_name=target)

    vars_3_data_preparation = inspect.currentframe().f_locals

    return df


if __name__ == "__main__":
    function_kwargs = {
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
    data_preparation(
        function_kwargs=function_kwargs, target=change_case(DATASET.TARGET)
    )
