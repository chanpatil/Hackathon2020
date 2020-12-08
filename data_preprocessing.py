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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ############################ Utility Functions ###########################
def numerical_imputation(df_numeric, strategy, numerical_cols):

    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imp.fit(df_numeric[numerical_cols])
    imp.transform(df_numeric[numerical_cols])  

    return df_numeric

def deal_numerical_data(df_numeric, numerical_cols):
    shapiro_test = stats.shapiro(df_numeric[numerical_cols])
    p_value = shapiro_test.pvalue
    print("Value",p_value)
    if p_value > 0.005:
        strategy = "mean"
    else:
        strategy = "median"

    print(strategy ,"is a used for Imputation of numerical columms")
    df_imputed_numeric = numerical_imputation(df_numeric, strategy, numerical_cols)
    return df_imputed_numeric

def deal_categorical_data(df_category, categorical_cols):

    df_category = pd.get_dummies(df_category[categorical_cols])
    col_list = df_category.columns.tolist()
    print("Category list:",col_list)

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
        df_numeric = dataset_dup.copy()
        df_nan = deal_numerical_data(df_numeric, numerical_cols)
        #print("df_imputed_numerics:\n",df_imputed_numerics)
    
    # Dealing with Categorical columns
    if len(categorical_cols) != 0:

        df_category = df_nan.copy()
        df_nan = deal_categorical_data(df_category, categorical_cols)
        df_nan = df_nan.replace(np.nan, "NaN") 
 
        #print("df_imputed_categorical:\n",df_imputed_categorical)

    return df_nan


def feature_transformation(dataset_nan):
    """
    performs the feature transformation of dataset_nan
    """

    scaler = StandardScaler()
    df_columns = dataset_nan.columns.to_list()
    scaled_data = scaler.fit_transform(dataset_nan)
    transformed_data = pd.DataFrame(scaled_data, columns = df_columns)
    return transformed_data

