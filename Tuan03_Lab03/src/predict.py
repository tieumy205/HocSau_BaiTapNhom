from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "models" / "sklearn" / "best_model.pkl"
SUBMISSION_PATH = ROOT / "submissions" / "submission.csv"


def main():
    test = pd.read_csv(TEST_PATH)
    test_ids = test["Id"].copy()

    pipeline = joblib.load(MODEL_PATH)
    pred_log = pipeline.predict(test)
    pred_price = np.expm1(pred_log)
    pred_price = np.maximum(pred_price, 0)

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": pred_price
    })

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
