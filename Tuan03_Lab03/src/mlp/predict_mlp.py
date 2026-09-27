# src/mlp/predict_mlp.py

from pathlib import Path

import sys
import json
import joblib

import numpy as np
import pandas as pd

import torch


# ============================================================
# 1. PROJECT PATH
# ============================================================

MLP_DIR = Path(
    __file__
).resolve().parent

SRC_DIR = MLP_DIR.parent

ROOT_DIR = SRC_DIR.parent


if str(
    SRC_DIR
) not in sys.path:

    sys.path.append(
        str(SRC_DIR)
    )


# ============================================================
# 2. IMPORT MODEL
# ============================================================

from mlp.model import (
    HousePriceMLP
)


# ============================================================
# 3. PATHS
# ============================================================

TEST_PATH = (
    ROOT_DIR
    / "data"
    / "test.csv"
)


MODEL_DIR = (
    ROOT_DIR
    / "models"
    / "mlp"
)


MODEL_PATH = (
    MODEL_DIR
    / "best_mlp.pt"
)


PREPROCESSOR_PATH = (
    MODEL_DIR
    / "mlp_preprocessor.pkl"
)


METADATA_PATH = (
    MODEL_DIR
    / "mlp_metadata.json"
)


SUBMISSION_PATH = (
    ROOT_DIR
    / "submissions"
    / "submission_mlp.csv"
)


# ============================================================
# 4. DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# 5. SPARSE → DENSE
# ============================================================

def to_dense_float32(
    X
):

    if hasattr(
        X,
        "toarray"
    ):

        X = X.toarray()


    return np.asarray(

        X,

        dtype=np.float32

    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [

        MODEL_PATH,

        PREPROCESSOR_PATH,

        METADATA_PATH

    ]


    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(

                f"Không tìm thấy:\n"
                f"{path}\n\n"
                f"Hãy chạy trước:\n"
                f"python src/mlp/train_mlp.py"

            )


    # ========================================================
    # STEP 1
    # LOAD TEST
    # ========================================================

    test = pd.read_csv(
        TEST_PATH
    )


    ids = test[
        "Id"
    ].copy()


    # ========================================================
    # STEP 2
    # LOAD FEATURE PIPELINE
    # ========================================================

    feature_pipeline = joblib.load(

        PREPROCESSOR_PATH

    )


    # ========================================================
    # STEP 3
    # TRANSFORM TEST
    # ========================================================

    X_test = (

        feature_pipeline
        .transform(
            test
        )

    )


    X_test = to_dense_float32(

        X_test

    )


    print(

        "Test processed:",

        X_test.shape

    )


    # ========================================================
    # STEP 4
    # LOAD METADATA
    # ========================================================

    with open(

        METADATA_PATH,

        "r",

        encoding="utf-8"

    ) as f:


        metadata = json.load(
            f
        )


    # ========================================================
    # STEP 5
    # BUILD MODEL
    # ========================================================

    device = get_device()


    model = HousePriceMLP(

        input_dim=int(
            metadata[
                "input_dim"
            ]
        ),

        hidden_dims=tuple(
            metadata[
                "hidden_dims"
            ]
        ),

        dropout=float(
            metadata[
                "dropout"
            ]
        )

    ).to(
        device
    )


    # ========================================================
    # STEP 6
    # LOAD WEIGHTS
    # ========================================================

    state_dict = torch.load(

        MODEL_PATH,

        map_location=device

    )


    model.load_state_dict(

        state_dict

    )


    model.eval()


    # ========================================================
    # STEP 7
    # PREDICT
    # ========================================================

    X_tensor = torch.tensor(

        X_test,

        dtype=torch.float32

    ).to(
        device
    )


    with torch.no_grad():

        pred_log = model(

            X_tensor

        ).cpu().numpy()


    # ========================================================
    # STEP 8
    # log → SalePrice
    # ========================================================

    pred_price = np.expm1(

        pred_log

    )


    pred_price = np.maximum(

        pred_price,

        0

    )


    # ========================================================
    # STEP 9
    # CREATE SUBMISSION
    # ========================================================

    submission = pd.DataFrame({

        "Id":
            ids,

        "SalePrice":
            pred_price

    })


    SUBMISSION_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    submission.to_csv(

        SUBMISSION_PATH,

        index=False

    )


    print(

        "\nSubmission saved to:"

    )


    print(

        SUBMISSION_PATH

    )


    print(
        "\nPreview:"
    )


    print(

        submission.head()

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()