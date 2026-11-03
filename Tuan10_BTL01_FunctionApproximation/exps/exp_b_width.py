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

def run_experiment_b():
    device = get_device()
    cfg = config.BASELINE_CONFIG
    widths = list(range(1, 11)) # B = 1, 2, ..., 10
    
    records = []
    print("\n--- BẮT ĐẦU THỰC NGHIỆM NHÓM B (ĐỘ RỘNG TẦNG ẨN) ---")
    
    for B in widths:
        seed_metrics = {"train_mse": [], "val_mse": [], "test_mse": [], "test_mae": [], "test_rmse": []}
        
        for seed in cfg["seeds"]:
            set_seed(seed)
            train_loader, val_loader, test_loader = create_dataloaders(
                train_size=cfg["train_size"], batch_size=cfg["batch_size"]
            )
            
            model = FlexibleMLP(
                input_dim=2, hidden_layers=[B], 
                output_dim=1, activation=cfg["activation"]
            )
            
            ckpt_path = f"checkpoints/exp_b_B{B}_seed{seed}.pt"
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
            "B_num_neurons": B,
            "train_mse_mean": np.mean(seed_metrics["train_mse"]),
            "train_mse_std": np.std(seed_metrics["train_mse"]),
            "val_mse_mean": np.mean(seed_metrics["val_mse"]),
            "val_mse_std": np.std(seed_metrics["val_mse"]),
            "test_mse_mean": np.mean(seed_metrics["test_mse"]),
            "test_mse_std": np.std(seed_metrics["test_mse"]),
            "test_mae_mean": np.mean(seed_metrics["test_mae"]),
            "test_rmse_mean": np.mean(seed_metrics["test_rmse"]),
        }
        records.append(row)
        print(f"B = {B:2d} nơ-ron | Train MSE: {row['train_mse_mean']:.6f} | Val MSE: {row['val_mse_mean']:.6f} | Test MSE: {row['test_mse_mean']:.6f}")

    os.makedirs("logs", exist_ok=True)
    df_res = pd.DataFrame(records)
    df_res.to_csv("logs/results_group_b.csv", index=False)
    
    # Vẽ đồ thị đánh giá Underfitting / Overfitting theo B
    plt.figure(figsize=(9, 5))
    plt.errorbar(df_res["B_num_neurons"], df_res["train_mse_mean"], yerr=df_res["train_mse_std"], label="Train MSE", marker='o', capsize=4)
    plt.errorbar(df_res["B_num_neurons"], df_res["val_mse_mean"], yerr=df_res["val_mse_std"], label="Validation MSE", marker='s', capsize=4)
    plt.errorbar(df_res["B_num_neurons"], df_res["test_mse_mean"], yerr=df_res["test_mse_std"], label="Test MSE", marker='^', capsize=4)
    plt.xlabel("Số nơ-ron trong 1 tầng ẩn (B)")
    plt.ylabel("Sai số MSE (Mean ± Std)")
    plt.title("Thực nghiệm Nhóm B: Tác động của độ rộng tầng ẩn")
    plt.xticks(widths)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/group_b_width_curve.png", dpi=300)
    plt.close()
    return df_res