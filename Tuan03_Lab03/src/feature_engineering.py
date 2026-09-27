import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _series(df, col):
    return df[col].fillna(0) if col in df.columns else pd.Series(0, index=df.index)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalSF"] = _series(df, "TotalBsmtSF") + _series(df, "1stFlrSF") + _series(df, "2ndFlrSF")
    df["Total_sqr_footage"] = (
        _series(df, "BsmtFinSF1") + _series(df, "BsmtFinSF2")
        + _series(df, "1stFlrSF") + _series(df, "2ndFlrSF")
    )
    df["TotalBsmtAndFirstFlrSF"] = _series(df, "TotalBsmtSF") + _series(df, "1stFlrSF")

    df["TotalBathrooms"] = (
        _series(df, "FullBath") + 0.5 * _series(df, "HalfBath")
        + _series(df, "BsmtFullBath") + 0.5 * _series(df, "BsmtHalfBath")
    )

    if "TotRmsAbvGrd" in df.columns:
        df["TotalRooms"] = df["TotRmsAbvGrd"]
    if "BedroomAbvGr" in df.columns:
        df["TotalBedrooms"] = df["BedroomAbvGr"]

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    if {"YrSold", "GarageYrBlt"}.issubset(df.columns):
        df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)

    if {"GarageArea", "GarageCars"}.issubset(df.columns):
        cars = df["GarageCars"].fillna(0)
        area = df["GarageArea"].fillna(0)
        df["GarageAreaPerCar"] = np.where(cars > 0, area / cars, 0)

    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
    existing = [c for c in porch_cols if c in df.columns]
    if existing:
        df["TotalPorchSF"] = df[existing].fillna(0).sum(axis=1)

    if {"OverallQual", "GrLivArea"}.issubset(df.columns):
        df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    if "OverallQual" in df.columns and "TotalSF" in df.columns:
        df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    if {"GrLivArea", "GarageArea"}.issubset(df.columns):
        df["LivingAndGarageArea"] = _series(df, "GrLivArea") + _series(df, "GarageArea")
    if {"GrLivArea", "TotalBsmtSF"}.issubset(df.columns):
        df["LivingAndBasementArea"] = _series(df, "GrLivArea") + _series(df, "TotalBsmtSF")

    boolean_sources = {
        "HasGarage": "GarageArea", "HasBasement": "TotalBsmtSF", "HasFireplace": "Fireplaces",
        "Has2ndFloor": "2ndFlrSF", "HasMasonryVeneer": "MasVnrArea",
        "HasWoodDeck": "WoodDeckSF", "HasOpenPorch": "OpenPorchSF"
    }
    for new_col, source_col in boolean_sources.items():
        if source_col in df.columns:
            df[new_col] = (_series(df, source_col) > 0).astype(int)

    return df


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering so the same transformations run in CV and prediction."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return create_features(X)
