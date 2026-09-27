# src/preprocessing.py

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# xóa cột ID. Vì feature này không cần thiết

def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()


    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    return df



# các feature này NaN thì nó có nghĩa là không có. Nên đặt cho nó giá trị là None
def fill_missing_as_none(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    none_cols = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",

        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",

        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",

        "MasVnrType"
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df



# các feature này NaN, nghĩa là không có, nên fillna = 0
def fill_missing_as_zero(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    zero_cols = [
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",

        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",

        "BsmtFullBath",
        "BsmtHalfBath",

        "MasVnrArea"
    ]

    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df



# điền giá trị median giữa các lotFrontage cùng Neighborhood
def fill_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:

        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
            .transform(
                lambda x: x.fillna(x.median())
            )
        )

# trường hợp LotFrontage không có Neighborhood
        df["LotFrontage"] = df["LotFrontage"].fillna(
            df["LotFrontage"].median()
        )

    return df

# Lấy mode 
def fill_electrical(df: pd.DataFrame) ->pd.DataFrame:
    df = df.copy()
    
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    
    return df

# đối với các numeric feature nhưng mang ý nghĩa categorical thì chuyển sang dạng categorical
def convert_special_categorical_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    

    df = df.copy()

    categorical_numeric_cols = [
        "MSSubClass",
        "MoSold",
        "YrSold"
    ]

    for col in categorical_numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df




def clean_data(df: pd.DataFrame) -> pd.DataFrame:


    df = df.copy()

    # 1. xóa ID
    df = remove_unnecessary_columns(df)

    # 2. NaN -> "None" cho các feature không tồn tại
    df = fill_missing_as_none(df)

    # 3. NaN -> 0 cho các numerical feature không tồn tại
    df = fill_missing_as_zero(df)

    # 4. LotFrontage
    df = fill_lot_frontage(df)

    # 5. Chuyển Numeric -> categorical feature
    df = convert_special_categorical_columns(df)
    
    #6. Feature Electrical = mode
    df = fill_electrical(df)

    return df




def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = True
) -> ColumnTransformer:


    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()



    numerical_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    if scale_numeric:
        numerical_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )



    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )



    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_cols
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_cols
            )
        ],
        remainder="drop"
    )

    return preprocessor




def prepare_train_data(
    train_df: pd.DataFrame,
    target_col: str = "SalePrice"
):


    df = train_df.copy()

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    
    y = df[target_col].copy()

    
    X = df.drop(columns=[target_col])

    
    X = clean_data(X)

    return X, y




def prepare_test_data(
    test_df: pd.DataFrame
) -> pd.DataFrame:


    X_test = test_df.copy()

    X_test = clean_data(X_test)

    return X_test




def transform_target(y: pd.Series) -> pd.Series:


    return np.log1p(y)




def inverse_transform_target(
    y_pred: np.ndarray
) -> np.ndarray:
    

    return np.expm1(y_pred)




def preprocess_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_numeric: bool = True
):


    X_train, y = prepare_train_data(train_df)



    X_test = prepare_test_data(test_df)



    preprocessor = build_preprocessor(
        X_train,
        scale_numeric=scale_numeric
    )



    X_train_processed = preprocessor.fit_transform(
        X_train
    )


    X_test_processed = preprocessor.transform(
        X_test
    )

    return (
        X_train_processed,
        X_test_processed,
        y,
        preprocessor
    )