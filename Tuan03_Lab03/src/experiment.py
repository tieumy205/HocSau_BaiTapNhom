# src/experiment.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.pipeline import Pipeline

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
    build_feature_selector
)


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/train.csv"

RESULT_PATH = "exps/model_feature_experiments.csv"

TEST_SIZE = 0.2

N_SPLITS = 5


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    return X, y


# ============================================================
# 3. DEFINE MODELS
# ============================================================

def get_models():

    models = {

        # -----------------------------------------
        # Ridge Regression
        # -----------------------------------------

        "Ridge": Ridge(
            alpha=10.0
        ),

        # -----------------------------------------
        # ElasticNet
        # -----------------------------------------

        "ElasticNet": ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=RANDOM_STATE
        ),

        # -----------------------------------------
        # Random Forest
        # -----------------------------------------

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        # -----------------------------------------
        # Gradient Boosting
        # -----------------------------------------

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================
# 4. BUILD PIPELINE
# ============================================================

def build_pipeline(
    model,
    k="all",
    use_feature_engineering=True
):

    steps = [

        (
            "cleaner",
            HouseDataCleaner()
        )

    ]

    # Feature Engineering
    if use_feature_engineering:

        steps.append(
            (
                "feature_engineering",
                HouseFeatureEngineer()
            )
        )

    # Preprocessing
    steps.append(
        (
            "preprocessor",
            build_preprocessor()
        )
    )

    # Feature Selection
    steps.append(
        (
            "feature_selection",
            build_feature_selector(
                k=k
            )
        )
    )

    # Model
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
# 5. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    model_name,
    model,
    X,
    y_log,
    k,
    use_feature_engineering,
    cv
):

    pipeline = build_pipeline(
        model=model,
        k=k,
        use_feature_engineering=use_feature_engineering
    )

    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse_scores = -scores

    result = {

        "model": model_name,

        "feature_engineering":
            use_feature_engineering,

        "k":
            k,

        "cv_rmse_mean":
            rmse_scores.mean(),

        "cv_rmse_std":
            rmse_scores.std(),

        "fold_1":
            rmse_scores[0],

        "fold_2":
            rmse_scores[1],

        "fold_3":
            rmse_scores[2],

        "fold_4":
            rmse_scores[3],

        "fold_5":
            rmse_scores[4]
    }

    return result


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Hold-out validation
    #
    # Experiment chỉ dùng X_dev.
    # X_valid để dành cho train.py đánh giá cuối.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Transform target
    # --------------------------------------------------------

    y_dev_log = np.log1p(
        y_dev
    )

    # --------------------------------------------------------
    # Cross Validation
    # --------------------------------------------------------

    cv = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = get_models()

    # --------------------------------------------------------
    # Feature numbers
    # --------------------------------------------------------

    k_values = [

        20,

        40,

        60,

        80,

        100,

        "all"
    ]

    # --------------------------------------------------------
    # Save results here
    # --------------------------------------------------------

    results = []

    # ========================================================
    # BASELINE
    #
    # Không Feature Engineering
    # Không Feature Selection
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "BASELINE EXPERIMENTS"
    )

    print(
        "=============================="
    )

    for model_name, model in models.items():

        print(
            f"\nRunning baseline: {model_name}"
        )

        result = run_experiment(

            model_name=model_name,

            model=model,

            X=X_dev,

            y_log=y_dev_log,

            k="all",

            use_feature_engineering=False,

            cv=cv
        )

        results.append(
            result
        )

        print(
            f"CV RMSE = "
            f"{result['cv_rmse_mean']:.5f}"
        )

    # ========================================================
    # FEATURE ENGINEERING + FEATURE SELECTION
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "FEATURE EXPERIMENTS"
    )

    print(
        "=============================="
    )

    for model_name, model in models.items():

        for k in k_values:

            print(
                f"\nModel={model_name}"
                f" | k={k}"
            )

            result = run_experiment(

                model_name=model_name,

                model=model,

                X=X_dev,

                y_log=y_dev_log,

                k=k,

                use_feature_engineering=True,

                cv=cv
            )

            results.append(
                result
            )

            print(
                f"Mean CV RMSE = "
                f"{result['cv_rmse_mean']:.5f}"
            )

            print(
                f"Std = "
                f"{result['cv_rmse_std']:.5f}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # Sort lowest RMSE first
    results_df = results_df.sort_values(
        by="cv_rmse_mean",
        ascending=True
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            RESULT_PATH
        ),
        exist_ok=True
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Show best experiments
    # --------------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "TOP EXPERIMENTS"
    )

    print(
        "===================================="
    )

    print(
        results_df[
            [
                "model",
                "feature_engineering",
                "k",
                "cv_rmse_mean",
                "cv_rmse_std"
            ]
        ].head(15)
    )

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    best = results_df.iloc[0]

    print(
        "\n===================================="
    )

    print(
        "BEST CONFIGURATION"
    )

    print(
        "===================================="
    )

    print(
        f"Model: "
        f"{best['model']}"
    )

    print(
        f"Feature Engineering: "
        f"{best['feature_engineering']}"
    )

    print(
        f"k: "
        f"{best['k']}"
    )

    print(
        f"Mean CV RMSE: "
        f"{best['cv_rmse_mean']:.5f}"
    )

    print(
        f"Std CV RMSE: "
        f"{best['cv_rmse_std']:.5f}"
    )

    print(
        f"\nResults saved to: "
        f"{RESULT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()