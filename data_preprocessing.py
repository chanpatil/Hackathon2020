"""
dataset_dup = dp_dup.remove_duplicate_records(dataset)
dataset_nan = dp_nan.deal_with_nan(dataset_dup)
dataset_cat = dp_cat.deal_with_categorical_data(dataset_nan)
dataset_out = dp_out.deal_with_outlier(dataset_cat)
dataset_scale = dp_scale.scale_feature(dataset_out)
"""
import pandas as pd
import numpy as np
from scipy import stats
#from sklearn.impute import SimpleImputer
#from sklearn_pandas import CategoricalImputer

# ############################ Utility Functions ###########################
def numerical_imputation(df_numeric, strategy):

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(df_numeric)
    imp.transform(df_numeric)  

    return df_numeric

def deal_numerical_data(df_numeric):
    shapiro_test = stats.shapiro(df_numeric)
    p_value = shapiro_test.pvalue
    print("Value",p_value)
    if p_value > 0.005:
        strategy = "mean"
    else:
        strategy = "median"

    print(strategy ,"is a used for Imputation of numerical columms")
    df_imputed_numeric = numerical_imputation(df_numeric, strategy)
    return df_imputed_numeric

def deal_categorical_data(df_category):
    cat_imputer = CategoricalImputer()
    cat_imputer.fit_transform(df_category)
    return df_category

# ######################## Parent Functions ########################


def remove_duplicate_records(dataset):
    
    # dropping duplicate values
    print("**************Inside of remove_duplicate_records()***************")
    print(dataset.shape)
    dataset.drop_duplicates(keep=False,inplace=True) 
    print(dataset.shape)
    
    return dataset

def deal_with_nan(dataset_dup):
    """
    Need to handle for the following scenario:
        1. dealing NaN in numerical columns
        2. dealing NaN in categorical columns
    """
    print("**************Inside of deal_with_nan()***************")
    # List containing numerical features only
    numerical_cols = dataset_dup.select_dtypes(include=['int', 'float']).columns.tolist()
    print("Numerical Cols:", numerical_cols)
    # List containing categorical features only
    categorical_cols = dataset_dup.select_dtypes(exclude=['int', 'float','datetime64']).columns.to_list()
    print("Categorical Cols:", categorical_cols)
    # List containing DateTime features only
    time_series = dataset_dup.select_dtypes(include=['datetime64']).columns.to_list()
    print("Time Series Cols:", time_series)
    """
    print(numerical_cols)
    print(categorical_cols)
    print(time_series)
    """
    print(dataset_dup.shape)
    # Dealing with the Timeseries columns
    if len(time_series) != 0:
        dataset_dup.drop(time_series, axis=1)

    # Dealing with numeric Columns
    if len(numerical_cols) != 0:
        df_numeric = dataset_dup[numerical_cols].copy()
        df_imputed_numerics = deal_numerical_data(df_numeric)
        print(df_imputed_numerics)
    
    # Dealing with Categorical columns
    if len(categorical_cols) != 0:
        df_category = dataset_dup[categorical_cols].copy()
        df_imputed_categorical = deal_categorical_data(df_category)
        print(df_imputed_categorical)

    # Merge numerical and categorical data
    preprocessed_df = pd.concat([df_imputed_numerics, df_imputed_categorical], axis=1)
    print(preprocessed_df)
    print(preprocessed_df.shape)

    # Dealing with Categorical Columns
    


    print("******************************************************************")
    return "Work in Progress"
