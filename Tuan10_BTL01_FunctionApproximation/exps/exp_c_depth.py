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

def run_experiment_c():
    device = get_device()
    cfg = config.BASELINE_CONFIG
    base_width = cfg["hidden_layers"][0]
    depths = [1, 2, 3, 4] 
    
    records = []
    convergence_curves = {}
    print("\n--- BẮT ĐẦU THỰC NGHIỆM NHÓM C (ĐỘ SÂU MẠNG & THỜI GIAN) ---")
    
    for C in depths:
        hidden_arch = [base_width] * C # Ví dụ C=3 -> [8, 8, 8]
        seed_metrics = {"train_mse": [], "val_mse": [], "test_mse": [], "test_mae": [], "test_rmse": [], "time": []}
        val_histories = []
        
        for seed in cfg["seeds"]:
            set_seed(seed)
            train_loader, val_loader, test_loader = create_dataloaders(
                train_size=cfg["train_size"], batch_size=cfg["batch_size"]
            )
            
            model = FlexibleMLP(
                input_dim=2, hidden_layers=hidden_arch, 
                output_dim=1, activation=cfg["activation"]
            )
            
            ckpt_path = f"checkpoints/exp_c_C{C}_seed{seed}.pt"
            model, history, _, _, elapsed = train_model(
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
            seed_metrics["time"].append(elapsed)
            val_histories.append(history["val_loss"])
            
        convergence_curves[C] = np.mean(val_histories, axis=0)
        
        row = {
            "C_num_layers": C,
            "architecture": str(hidden_arch),
            "train_mse_mean": np.mean(seed_metrics["train_mse"]),
            "val_mse_mean": np.mean(seed_metrics["val_mse"]),
            "val_mse_std": np.std(seed_metrics["val_mse"]),
            "test_mse_mean": np.mean(seed_metrics["test_mse"]),
            "test_mse_std": np.std(seed_metrics["test_mse"]),
            "test_mae_mean": np.mean(seed_metrics["test_mae"]),
            "test_rmse_mean": np.mean(seed_metrics["test_rmse"]),
            "train_time_sec_mean": np.mean(seed_metrics["time"]),
            "train_time_sec_std": np.std(seed_metrics["time"]),
        }
        records.append(row)
        print(f"C = {C} tầng {str(hidden_arch):14s} | Val MSE: {row['val_mse_mean']:.6f} | Test MSE: {row['test_mse_mean']:.6f} | Time: {row['train_time_sec_mean']:.2f}s")

    os.makedirs("logs", exist_ok=True)
    df_res = pd.DataFrame(records)
    df_res.to_csv("logs/results_group_c.csv", index=False)
    
    # Vẽ biểu đồ kép: (1) Tốc độ hội tụ Val Loss theo Epoch và (2) Sai số vs Thời gian huấn luyện
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Đồ thị 1: Tốc độ hội tụ
    for C, curve in convergence_curves.items():
        ax1.plot(range(1, len(curve) + 1), curve, label=f"C = {C} tầng ẩn")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MSE Loss (Log Scale)")
    ax1.set_title("Tốc độ hội tụ Loss theo độ sâu C")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Đồ thị 2: Test Error & Thời gian huấn luyện
    ax2_time = ax2.twinx()
    p1 = ax2.errorbar(df_res["C_num_layers"], df_res["test_mse_mean"], yerr=df_res["test_mse_std"], 
                      color='tab:blue', marker='o', capsize=4, label="Test MSE")
    p2 = ax2_time.bar(df_res["C_num_layers"], df_res["train_time_sec_mean"], alpha=0.3, 
                      color='tab:orange', width=0.4, label="Thời gian train (s)")
    
    ax2.set_xlabel("Số tầng ẩn (C)")
    ax2.set_ylabel("Test MSE", color='tab:blue')
    ax2_time.set_ylabel("Thời gian huấn luyện (Giây)", color='tab:orange')
    ax2.set_xticks(depths)
    ax2.set_title("Đánh đổi giữa Sai số Test và Thời gian huấn luyện")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("figures/group_c_depth_time.png", dpi=300)
    plt.close()
    return df_res