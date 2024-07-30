# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:05:39 2024

@author: Perat
"""
from functools import reduce

import numpy as np
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from scipy import stats


def change_case(string: str):
    """
    Changes the case of a given string to snake case by replacing spaces with underscores and upper case to lower case.

    Parameters:
    string (str): The input string to be modified.

    Returns:
    str: The modified string with case changed and spaces replaced by underscores.
    """
    # Using reduce to concatenate characters and add underscores before uppercase letters
    modified_string = reduce(lambda x, y: x + ("_" if y.isupper() else "") + y, string)
    # Replacing spaces with underscores and removing any double underscores
    final_string = modified_string.replace(" ", "_").replace("__", "_").lower()

    return final_string


def dataframe_detail_describe(df: DataFrame, target_name: str) -> DataFrame:
    """
    Generate a detailed statistical summary of the DataFrame and calculate additional columns for analysis.

    Parameters:
    - df (DataFrame): The input DataFrame for which the details are to be described.
    - target_name (str): The name of the target column for outlier analysis.

    Returns:
    - DataFrame: A DataFrame containing detailed statistical information and additional columns for analysis.
    """

    # Generate basic statistics for each column and transpose the table
    info_table = df.describe(
        include="all",
        percentiles=[],
    ).T.drop(
        ["unique", "top", "freq"],
        axis=1,
    )

    # Add a column for data types
    info_table["Dtype"] = df.dtypes

    # Reset the index and rename the column
    info_table.reset_index(inplace=True)
    info_table.rename(columns={"index": "column_name"}, inplace=True)

    # Calculate the count of null values and completeness percentage for each column
    info_table["null_count"] = info_table["count"].apply(lambda x: df.shape[0] - x)
    info_table["completeness"] = info_table["count"].apply(
        lambda x: x / df.shape[0] * 100
    )

    # Identify extreme outliers in each numeric column (excluding the target column)
    outlier_extreme = []
    for column_name in df.columns:
        if is_numeric_dtype(df[column_name]) and column_name != target_name:
            z = np.abs(stats.zscore(df[column_name], nan_policy="omit"))
            threshold_z = 3
            outlier_indices = np.where(z > threshold_z)[0]
            outlier_extreme.append(len(outlier_indices))
        else:
            outlier_extreme.append(np.nan)

    # Add a column for the count of extreme outliers
    info_table["outlier_extreme_count"] = outlier_extreme

    return info_table


def rename_columns(df: DataFrame):
    rename_mapping = {
        column_name: change_case(string=column_name) for column_name in df.columns
    }
    df.rename(columns=rename_mapping, inplace=True)
