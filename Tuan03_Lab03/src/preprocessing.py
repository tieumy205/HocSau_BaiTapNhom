import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType"
]

ZERO_COLS = [
    "GarageYrBlt", "GarageCars", "GarageArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
]

SPECIAL_CATEGORICAL_COLS = ["MSSubClass"]


class HouseDataCleaner(BaseEstimator, TransformerMixin):
    """House Prices-specific deterministic cleaning."""

    def __init__(self, drop_id=True):
        self.drop_id = drop_id

    def fit(self, X, y=None):
        X = X.copy()
        if "LotFrontage" in X.columns:
            self.lot_frontage_global_median_ = X["LotFrontage"].median()
            if "Neighborhood" in X.columns:
                self.lot_frontage_by_neighborhood_ = X.groupby("Neighborhood")["LotFrontage"].median()
            else:
                self.lot_frontage_by_neighborhood_ = None
        else:
            self.lot_frontage_global_median_ = None
            self.lot_frontage_by_neighborhood_ = None
        return self

    def transform(self, X):
        X = X.copy()

        if self.drop_id and "Id" in X.columns:
            X = X.drop(columns=["Id"])

        for col in NONE_COLS:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        for col in ZERO_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        if "LotFrontage" in X.columns:
            if "Neighborhood" in X.columns and self.lot_frontage_by_neighborhood_ is not None:
                mapped = X["Neighborhood"].map(self.lot_frontage_by_neighborhood_)
                X["LotFrontage"] = X["LotFrontage"].fillna(mapped)
            X["LotFrontage"] = X["LotFrontage"].fillna(self.lot_frontage_global_median_)

        for col in SPECIAL_CATEGORICAL_COLS:
            if col in X.columns:
                X[col] = X[col].astype(str)

        return X


def build_preprocessor(scale_numeric=True):
    numerical_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_pipeline = Pipeline(numerical_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, selector(dtype_include="number")),
            ("cat", categorical_pipeline, selector(dtype_include=["object", "category"]))
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
