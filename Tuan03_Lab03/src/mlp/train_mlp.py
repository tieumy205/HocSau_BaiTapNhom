# src/mlp/train_mlp.py

from pathlib import Path

import sys
import copy
import json
import joblib

import numpy as np
import pandas as pd

import torch

from torch import nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

MLP_DIR = Path(
    __file__
).resolve().parent

SRC_DIR = MLP_DIR.parent

ROOT_DIR = SRC_DIR.parent


# Cho phép import file trong src/
if str(SRC_DIR) not in sys.path:

    sys.path.append(
        str(SRC_DIR)
    )


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

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

from mlp.model import (
    HousePriceMLP
)


# ============================================================
# 3. CONFIG
# ============================================================

RANDOM_STATE = 42


DATA_PATH = (
    ROOT_DIR
    / "data"
    / "train.csv"
)


EXPERIMENT_RESULT_PATH = (
    ROOT_DIR
    / "exps"
    / "model_feature_experiments.csv"
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


HISTORY_PATH = (
    MODEL_DIR
    / "training_history.csv"
)


# ------------------------------------------------------------
# Training config
# ------------------------------------------------------------

TEST_SIZE = 0.2

BATCH_SIZE = 64

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

MAX_EPOCHS = 300

PATIENCE = 30


HIDDEN_DIMS = (
    256,
    128,
    64
)

DROPOUT = 0.2


# ============================================================
# 4. RANDOM SEED
# ============================================================

def set_seed(
    seed=RANDOM_STATE
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 5. DEVICE
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
# 6. LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        columns=["SalePrice"]
    )

    y = df[
        "SalePrice"
    ].astype(float)

    return X, y


# ============================================================
# 7. LOAD FEATURE CONFIGURATION
# ============================================================

def load_feature_configuration():
    """
    Dùng lại Feature Engineering + k
    từ experiment sklearn tốt nhất.

    Như vậy MLP nhận cùng kiểu feature
    với nhánh sklearn.
    """

    if not EXPERIMENT_RESULT_PATH.exists():

        raise FileNotFoundError(

            f"Không tìm thấy:\n"
            f"{EXPERIMENT_RESULT_PATH}\n\n"
            f"Hãy chạy trước:\n"
            f"python src/experiment.py"
        )

    results = pd.read_csv(
        EXPERIMENT_RESULT_PATH
    )

    results = results.sort_values(
        by="cv_rmse_mean",
        ascending=True
    )

    best = results.iloc[0]


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    fe_value = best[
        "feature_engineering"
    ]

    if isinstance(
        fe_value,
        str
    ):

        use_feature_engineering = (
            fe_value.lower()
            == "true"
        )

    else:

        use_feature_engineering = bool(
            fe_value
        )


    # --------------------------------------------------------
    # k
    # --------------------------------------------------------

    k_value = best["k"]

    if str(
        k_value
    ).lower() == "all":

        k = "all"

    else:

        k = int(
            float(
                k_value
            )
        )


    return (
        use_feature_engineering,
        k,
        best
    )


# ============================================================
# 8. BUILD FEATURE PIPELINE
# ============================================================

def build_feature_pipeline(
    k,
    use_feature_engineering=True
):
    """
    Pipeline này chỉ xử lý feature.

    Không chứa model PyTorch.
    """

    steps = [

        (
            "cleaner",
            HouseDataCleaner()
        )

    ]


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    if use_feature_engineering:

        steps.append(

            (
                "feature_engineering",
                HouseFeatureEngineer()
            )

        )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    steps.append(

        (
            "preprocessor",
            build_preprocessor()
        )

    )


    # --------------------------------------------------------
    # Feature Selection
    # --------------------------------------------------------

    steps.append(

        (
            "feature_selection",
            build_feature_selector(
                k=k
            )
        )

    )


    return Pipeline(
        steps
    )


# ============================================================
# 9. SPARSE → DENSE
# ============================================================

def to_dense_float32(
    X
):
    """
    OneHotEncoder có thể tạo sparse matrix.

    PyTorch cần dense numpy float32.
    """

    if hasattr(
        X,
        "toarray"
    ):

        X = X.toarray()


    X = np.asarray(
        X,
        dtype=np.float32
    )

    return X


# ============================================================
# 10. CREATE DATALOADER
# ============================================================

def create_dataloader(
    X,
    y,
    batch_size,
    shuffle
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )


    dataset = TensorDataset(

        X_tensor,

        y_tensor

    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle

    )


    return loader


# ============================================================
# 11. EVALUATE
# ============================================================

def evaluate_model(
    model,
    data_loader,
    criterion,
    device
):

    model.eval()


    total_loss = 0.0


    predictions = []

    targets = []


    with torch.no_grad():

        for (
            X_batch,
            y_batch
        ) in data_loader:


            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )


            pred = model(
                X_batch
            )


            loss = criterion(

                pred,

                y_batch

            )


            total_loss += (

                loss.item()

                * X_batch.size(0)

            )


            predictions.append(

                pred
                .cpu()
                .numpy()

            )


            targets.append(

                y_batch
                .cpu()
                .numpy()

            )


    mean_loss = (

        total_loss

        / len(
            data_loader.dataset
        )

    )


    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )


    rmse = np.sqrt(

        mean_squared_error(

            targets,

            predictions

        )

    )


    return (
        mean_loss,
        rmse
    )


# ============================================================
# 12. TRAIN MLP
# ============================================================

def train_model(
    model,
    train_loader,
    valid_loader,
    device
):

    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    best_valid_rmse = float(
        "inf"
    )


    best_state = None


    patience_counter = 0


    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):


        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        train_loss_sum = 0.0


        for (
            X_batch,
            y_batch
        ) in train_loader:


            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )


            optimizer.zero_grad()


            pred = model(
                X_batch
            )


            loss = criterion(

                pred,

                y_batch

            )


            loss.backward()


            optimizer.step()


            train_loss_sum += (

                loss.item()

                * X_batch.size(0)

            )


        train_loss = (

            train_loss_sum

            / len(
                train_loader.dataset
            )

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            valid_loss,
            valid_rmse
        ) = evaluate_model(

            model,

            valid_loader,

            criterion,

            device

        )


        history.append({

            "epoch":
                epoch,

            "train_mse":
                train_loss,

            "valid_mse":
                valid_loss,

            "valid_rmse_log":
                valid_rmse

        })


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(

                f"Epoch {epoch:03d}"

                f" | Train MSE: "
                f"{train_loss:.6f}"

                f" | Valid RMSE: "
                f"{valid_rmse:.6f}"

            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if valid_rmse < best_valid_rmse:


            best_valid_rmse = (
                valid_rmse
            )


            best_state = copy.deepcopy(

                model.state_dict()

            )


            patience_counter = 0


        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:


            print(

                f"\nEarly stopping "
                f"at epoch {epoch}"

            )


            break


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )


    return (

        model,

        history,

        best_valid_rmse

    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed()


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()


    print(
        "Device:",
        device
    )


    # ========================================================
    # STEP 1
    # LOAD DATA
    # ========================================================

    X, y = load_data()


    print(
        "Dataset:",
        X.shape
    )


    # ========================================================
    # STEP 2
    # LOAD FEATURE CONFIG
    # ========================================================

    (
        use_feature_engineering,
        k,
        best_sklearn
    ) = load_feature_configuration()


    print(
        "\nFeature configuration:"
    )


    print(

        "Feature Engineering:",

        use_feature_engineering

    )


    print(
        "k:",
        k
    )


    print(

        "Best sklearn model:",

        best_sklearn["model"]

    )


    # ========================================================
    # STEP 3
    # TRAIN / VALID SPLIT
    # ========================================================

    (
        X_train,
        X_valid,
        y_train,
        y_valid
    ) = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )


    # ========================================================
    # STEP 4
    # FEATURE PIPELINE
    # ========================================================

    feature_pipeline = (

        build_feature_pipeline(

            k=k,

            use_feature_engineering=
                use_feature_engineering

        )

    )


    # ========================================================
    # STEP 5
    # FIT ONLY TRAIN
    # ========================================================

    X_train_processed = (

        feature_pipeline
        .fit_transform(

            X_train,

            np.log1p(
                y_train
            )

        )

    )


    # Validation chỉ transform
    X_valid_processed = (

        feature_pipeline.transform(
            X_valid
        )

    )


    # ========================================================
    # STEP 6
    # SPARSE → DENSE
    # ========================================================

    X_train_processed = (

        to_dense_float32(

            X_train_processed

        )

    )


    X_valid_processed = (

        to_dense_float32(

            X_valid_processed

        )

    )


    # ========================================================
    # STEP 7
    # TARGET LOG
    # ========================================================

    y_train_log = np.log1p(

        y_train.to_numpy()

    ).astype(
        np.float32
    )


    y_valid_log = np.log1p(

        y_valid.to_numpy()

    ).astype(
        np.float32
    )


    print(

        "\nTrain processed:",

        X_train_processed.shape

    )


    print(

        "Valid processed:",

        X_valid_processed.shape

    )


    # ========================================================
    # STEP 8
    # DATA LOADERS
    # ========================================================

    train_loader = (

        create_dataloader(

            X_train_processed,

            y_train_log,

            BATCH_SIZE,

            shuffle=True

        )

    )


    valid_loader = (

        create_dataloader(

            X_valid_processed,

            y_valid_log,

            BATCH_SIZE,

            shuffle=False

        )

    )


    # ========================================================
    # STEP 9
    # BUILD MLP
    # ========================================================

    input_dim = (

        X_train_processed
        .shape[1]

    )


    model = HousePriceMLP(

        input_dim=input_dim,

        hidden_dims=HIDDEN_DIMS,

        dropout=DROPOUT

    ).to(
        device
    )


    print(
        "\nMLP:"
    )


    print(
        model
    )


    # ========================================================
    # STEP 10
    # TRAIN
    # ========================================================

    (
        model,
        history,
        best_valid_rmse
    ) = train_model(

        model,

        train_loader,

        valid_loader,

        device

    )


    print(

        "\nBest Hold-out RMSE "
        "on log(SalePrice):",

        f"{best_valid_rmse:.5f}"

    )


    # ========================================================
    # STEP 11
    # SAVE
    # ========================================================

    MODEL_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # --------------------------------------------------------
    # Save PyTorch model weights
    # --------------------------------------------------------

    torch.save(

        model.state_dict(),

        MODEL_PATH

    )


    # --------------------------------------------------------
    # Save sklearn preprocessing pipeline
    # --------------------------------------------------------

    joblib.dump(

        feature_pipeline,

        PREPROCESSOR_PATH

    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    pd.DataFrame(

        history

    ).to_csv(

        HISTORY_PATH,

        index=False

    )


    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "input_dim":
            input_dim,

        "hidden_dims":
            list(
                HIDDEN_DIMS
            ),

        "dropout":
            DROPOUT,

        "feature_engineering":
            use_feature_engineering,

        "k":
            k,

        "random_state":
            RANDOM_STATE,

        "best_valid_rmse_log":
            float(
                best_valid_rmse
            )

    }


    with open(

        METADATA_PATH,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            metadata,

            f,

            indent=2,

            ensure_ascii=False

        )


    print(
        "\nSaved:"
    )


    print(
        MODEL_PATH
    )


    print(
        PREPROCESSOR_PATH
    )


    print(
        METADATA_PATH
    )


    print(
        HISTORY_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()