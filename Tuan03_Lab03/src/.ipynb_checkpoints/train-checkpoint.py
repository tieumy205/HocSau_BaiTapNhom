# src/train.py

import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from preprocessing import (
    HouseDataCleaner,
    build_preprocessor
)

from feature_engineering import (
    HouseFeatureEngineer
)

from feature_selection import (
    build_feature_selector,
    get_selected_feature_names
)


# ============================================================
# 1. CONFIG
# ============================================================
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

RANDOM_STATE = 42

DATA_PATH = "data/train.csv"

EXPERIMENT_RESULT_PATH = (
    ROOT_DIR/"exps"/"model_feature_experiments.csv"
)

MODEL_PATH = (
    ROOT_DIR/"models"/"sklearn"/"best_model.pkl"
)

SELECTED_FEATURES_PATH = (
    ROOT_DIR/"models"/"sklearn"/"selected_features.txt"
)

BEST_CONFIG_PATH = (
    ROOT_DIR/"models"/"sklearn/best_config.txt"
)

TEST_SIZE = 0.2


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        columns=["SalePrice"]
    )

    y = df["SalePrice"]

    return X, y


# ============================================================
# 3. LOAD BEST CONFIGURATION
# ============================================================

def load_best_configuration():
    """
    Đọc kết quả experiment và lấy experiment
    có CV RMSE trung bình thấp nhất.
    """

    if not EXPERIMENT_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {EXPERIMENT_RESULT_PATH}\n"
            f"Hãy chạy trước:\n"
            f"python src/experiment.py"
    )

    results = pd.read_csv(
        EXPERIMENT_RESULT_PATH
    )

    # Đảm bảo kết quả được sắp theo RMSE tăng dần
    results = results.sort_values(
        by="cv_rmse_mean",
        ascending=True
    )

    best = results.iloc[0]

    model_name = best["model"]

    # -----------------------------------------
    # Feature Engineering
    # -----------------------------------------

    feature_engineering_value = (
        best["feature_engineering"]
    )

    # CSV đôi khi đọc boolean thành string
    if isinstance(
        feature_engineering_value,
        str
    ):
        use_feature_engineering = (
            feature_engineering_value.lower()
            == "true"
        )
    else:
        use_feature_engineering = bool(
            feature_engineering_value
        )

    # -----------------------------------------
    # k
    # -----------------------------------------

    k_value = best["k"]

    # CSV có thể đọc tất cả thành string
    if str(k_value).lower() == "all":

        k = "all"

    else:

        # Có thể là "60", "60.0", ...
        k = int(
            float(k_value)
        )

    return (
        model_name,
        k,
        use_feature_engineering,
        best
    )


# ============================================================
# 4. BUILD MODEL
# ============================================================

def build_model(model_name):
    """
    Tạo model đúng với model đã dùng trong experiment.py.

    QUAN TRỌNG:
    Hyperparameters ở đây phải giống experiment.py.
    """

    if model_name == "Ridge":

        return Ridge(
            alpha=10.0
        )

    elif model_name == "ElasticNet":

        return ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=RANDOM_STATE
        )

    elif model_name == "RandomForest":

        return RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "GradientBoosting":

        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(
            f"Model không được hỗ trợ: "
            f"{model_name}"
        )


# ============================================================
# 5. BUILD FINAL PIPELINE
# ============================================================

def build_final_pipeline(
    model_name,
    k,
    use_feature_engineering
):

    model = build_model(
        model_name
    )

    steps = []

    # -----------------------------------------
    # Cleaning
    # -----------------------------------------

    steps.append(
        (
            "cleaner",
            HouseDataCleaner()
        )
    )

    # -----------------------------------------
    # Feature Engineering
    # -----------------------------------------

    if use_feature_engineering:

        steps.append(
            (
                "feature_engineering",
                HouseFeatureEngineer()
            )
        )

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------

    steps.append(
        (
            "preprocessor",
            build_preprocessor()
        )
    )

    # -----------------------------------------
    # Feature Selection
    # -----------------------------------------

    steps.append(
        (
            "feature_selection",
            build_feature_selector(
                k=k
            )
        )
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------

    steps.append(
        (
            "model",
            model
        )
    )

    return Pipeline(
        steps
    )


# ============================================================
# 6. EVALUATE HOLD-OUT DATA
# ============================================================

def evaluate_model(
    pipeline,
    X_valid,
    y_valid
):

    # Model đang dự đoán log1p(SalePrice)
    y_pred_log = pipeline.predict(
        X_valid
    )

    y_valid_log = np.log1p(
        y_valid
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid_log,
            y_pred_log
        )
    )

    return rmse


# ============================================================
# 7. SAVE SELECTED FEATURES
# ============================================================

def save_selected_features(
    pipeline
):

    selected_features = (
        get_selected_feature_names(
            pipeline
        )
    )

    SELECTED_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
)

    with open(
        SELECTED_FEATURES_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for feature in selected_features:

            f.write(
                str(feature) + "\n"
            )

    return selected_features


# ============================================================
# 8. SAVE BEST CONFIG
# ============================================================

def save_best_config(
    model_name,
    k,
    use_feature_engineering,
    cv_rmse,
    holdout_rmse
):

    BEST_CONFIG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
)

    with open(
        BEST_CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Model: {model_name}\n"
        )

        f.write(
            f"Feature Engineering: "
            f"{use_feature_engineering}\n"
        )

        f.write(
            f"k: {k}\n"
        )

        f.write(
            f"CV RMSE: "
            f"{cv_rmse:.5f}\n"
        )

        f.write(
            f"Hold-out RMSE: "
            f"{holdout_rmse:.5f}\n"
        )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    # ========================================================
    # STEP 1
    # Load data
    # ========================================================

    X, y = load_data()

    print(
        "Dataset:",
        X.shape
    )

    # ========================================================
    # STEP 2
    # Load best experiment
    # ========================================================

    (
        best_model_name,
        best_k,
        use_feature_engineering,
        best_result
    ) = load_best_configuration()

    print(
        "\n================================"
    )

    print(
        "BEST CONFIGURATION FROM EXPERIMENT"
    )

    print(
        "================================"
    )

    print(
        "Model:",
        best_model_name
    )

    print(
        "Feature Engineering:",
        use_feature_engineering
    )

    print(
        "k:",
        best_k
    )

    print(
        "CV RMSE:",
        f"{best_result['cv_rmse_mean']:.5f}"
    )

    # ========================================================
    # STEP 3
    # Create same hold-out split as experiment.py
    # ========================================================

    (
        X_dev,
        X_valid,
        y_dev,
        y_valid
    ) = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE
    )

    print(
        "\nDevelopment:",
        X_dev.shape
    )

    print(
        "Validation:",
        X_valid.shape
    )

    # ========================================================
    # STEP 4
    # Build final pipeline
    # ========================================================

    pipeline = build_final_pipeline(

        model_name=
            best_model_name,

        k=
            best_k,

        use_feature_engineering=
            use_feature_engineering
    )

    # ========================================================
    # STEP 5
    # Train only on development data
    # ========================================================

    y_dev_log = np.log1p(
        y_dev
    )

    pipeline.fit(
        X_dev,
        y_dev_log
    )

    # ========================================================
    # STEP 6
    # Final hold-out evaluation
    # ========================================================

    holdout_rmse = evaluate_model(

        pipeline,

        X_valid,

        y_valid
    )

    print(
        "\n================================"
    )

    print(
        "FINAL HOLD-OUT RESULT"
    )

    print(
        "================================"
    )

    print(
        "RMSE on log(SalePrice):",
        f"{holdout_rmse:.5f}"
    )

    # ========================================================
    # STEP 7
    # Train again using ALL train.csv
    # ========================================================

    print(
        "\nRetraining final model "
        "on full training data..."
    )

    final_pipeline = build_final_pipeline(

        model_name=
            best_model_name,

        k=
            best_k,

        use_feature_engineering=
            use_feature_engineering
    )

    y_full_log = np.log1p(
        y
    )

    final_pipeline.fit(
        X,
        y_full_log
    )

    # ========================================================
    # STEP 8
    # Save final pipeline
    # ========================================================

    MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

    joblib.dump(
        final_pipeline,
        MODEL_PATH
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    # ========================================================
    # STEP 9
    # Save selected features
    # ========================================================

    selected_features = (
        save_selected_features(
            final_pipeline
        )
    )

    print(
        "\nNumber of selected features:",
        len(selected_features)
    )

    print(
        "Selected features saved to:"
    )

    print(
        SELECTED_FEATURES_PATH
    )

    # ========================================================
    # STEP 10
    # Save configuration
    # ========================================================

    save_best_config(

        model_name=
            best_model_name,

        k=
            best_k,

        use_feature_engineering=
            use_feature_engineering,

        cv_rmse=
            float(
                best_result[
                    "cv_rmse_mean"
                ]
            ),

        holdout_rmse=
            holdout_rmse
    )

    print(
        "\nBest configuration saved to:"
    )

    print(
        BEST_CONFIG_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()