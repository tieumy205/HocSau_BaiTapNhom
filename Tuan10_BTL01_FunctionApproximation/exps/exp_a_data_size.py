import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from src.utils import set_seed, get_device
from src.dataset import create_dataloaders
from models.mlp import FlexibleMLP
from src.trainer import train_model
from src.metrics import evaluate_predictions

def run_experiment_a():
    device = get_device()
    cfg = config.BASELINE_CONFIG
    train_sizes = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    
    records = []
    print("\n--- BẮT ĐẦU THỰC NGHIỆM NHÓM A (QUY MÔ DỮ LIỆU) ---")
    
    for A in train_sizes:
        seed_metrics = {"train_mse": [], "val_mse": [], "test_mse": [], "test_mae": [], "test_rmse": []}
        
        for seed in cfg["seeds"]:
            set_seed(seed)
            train_loader, val_loader, test_loader = create_dataloaders(
                train_size=A, batch_size=cfg["batch_size"]
            )
            
            model = FlexibleMLP(
                input_dim=2, hidden_layers=cfg["hidden_layers"], 
                output_dim=1, activation=cfg["activation"]
            )
            
            ckpt_path = f"checkpoints/exp_a_A{A}_seed{seed}.pt"
            model, _, _, _, _ = train_model(
                model, train_loader, val_loader, epochs=cfg["epochs"],
                lr=cfg["lr"], optimizer_name=cfg["optimizer"],
                checkpoint_path=ckpt_path, device=device
            )
            
            tr_m, _, _ = evaluate_predictions(model, train_loader, device)
            va_m, _, _ = evaluate_predictions(model, val_loader, device)
            te_m, _, _ = evaluate_predictions(model, test_loader, device)
            
            seed_metrics["train_mse"].append(tr_m["mse"])
            seed_metrics["val_mse"].append(va_m["mse"])
            seed_metrics["test_mse"].append(te_m["mse"])
            seed_metrics["test_mae"].append(te_m["mae"])
            seed_metrics["test_rmse"].append(te_m["rmse"])
            
        row = {
            "A_train_size": A,
            "train_mse_mean": np.mean(seed_metrics["train_mse"]),
            "train_mse_std": np.std(seed_metrics["train_mse"]),
            "val_mse_mean": np.mean(seed_metrics["val_mse"]),
            "val_mse_std": np.std(seed_metrics["val_mse"]),
            "test_mse_mean": np.mean(seed_metrics["test_mse"]),
            "test_mse_std": np.std(seed_metrics["test_mse"]),
            "test_mae_mean": np.mean(seed_metrics["test_mae"]),
            "test_mae_std": np.std(seed_metrics["test_mae"]),
            "test_rmse_mean": np.mean(seed_metrics["test_rmse"]),
            "test_rmse_std": np.std(seed_metrics["test_rmse"]),
        }
        records.append(row)
        print(f"A = {A:4d} | Val MSE: {row['val_mse_mean']:.6f} ± {row['val_mse_std']:.6f} | Test RMSE: {row['test_rmse_mean']:.6f}")

    os.makedirs("logs", exist_ok=True)
    df_res = pd.DataFrame(records)
    df_res.to_csv("logs/results_group_a.csv", index=False)
    
    # Vẽ đồ thị sai số theo A
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.errorbar(df_res["A_train_size"], df_res["train_mse_mean"], yerr=df_res["train_mse_std"], label="Train MSE", marker='o', capsize=4)
    plt.errorbar(df_res["A_train_size"], df_res["val_mse_mean"], yerr=df_res["val_mse_std"], label="Validation MSE", marker='s', capsize=4)
    plt.errorbar(df_res["A_train_size"], df_res["test_mse_mean"], yerr=df_res["test_mse_std"], label="Test MSE", marker='^', capsize=4)
    plt.xlabel("Số lượng mẫu huấn luyện (A)")
    plt.ylabel("Sai số MSE (Mean ± Std)")
    plt.title("Thực nghiệm Nhóm A: Tác động của quy mô dữ liệu huấn luyện")
    plt.xticks(train_sizes)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/group_a_error_curve.png", dpi=300)
    plt.close()
    return df_res