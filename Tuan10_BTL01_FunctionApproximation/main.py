# import config
# from src.utils import set_seed, get_device
# from src.dataset import create_dataloaders
# from models.mlp import FlexibleMLP
# from src.trainer import train_model
# from src.metrics import evaluate_predictions

# if __name__ == "__main__":
#     device = get_device()
#     cfg = config.BASELINE_CONFIG
#     set_seed(cfg["seeds"][0])
    
#     print(f"Đang chạy kiểm tra Baseline trên thiết bị: {device}")
#     train_loader, val_loader, test_loader = create_dataloaders(
#         train_size=cfg["train_size"], batch_size=cfg["batch_size"]
#     )
    
#     model = FlexibleMLP(
#         input_dim=2, 
#         hidden_layers=cfg["hidden_layers"], 
#         output_dim=1, 
#         activation=cfg["activation"]
#     )
    
#     model, history, best_val, best_ep, elapsed = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         epochs=cfg["epochs"],
#         lr=cfg["lr"],
#         optimizer_name=cfg["optimizer"],
#         checkpoint_path="checkpoints/best_baseline.pt",
#         device=device
#     )
    
#     val_metrics, _, _ = evaluate_predictions(model, val_loader, device)
#     print(f"Hoàn tất sau {elapsed:.2f}s | Best Epoch: {best_ep}")
#     print(f"Validation Metrics -> MSE: {val_metrics['mse']:.6f} | MAE: {val_metrics['mae']:.6f} | RMSE: {val_metrics['rmse']:.6f}")



import os
# 1. Khắc phục lỗi xung đột OpenMP (libiomp5md.dll) trên Windows/Conda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
# 2. Chuyển sang backend 'Agg' để vẽ và lưu file ảnh trực tiếp, tránh lỗi QThreadStorage
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from data.generate_data import gaussian_2d, sample_dataset, plot_data_exploration
from src.utils import set_seed, get_device
from src.dataset import create_dataloaders
from models.mlp import FlexibleMLP
from src.trainer import train_model
from src.metrics import evaluate_predictions
from exps.exp_a_data_size import run_experiment_a
from exps.exp_b_width import run_experiment_b
from exps.exp_c_depth import run_experiment_c
from src.excel_logger import export_experiments_to_excel

def visualize_optimal_approximation(model, test_loader, device, save_path="figures/approximation_surface.png"):
    """Vẽ đồ thị so sánh bề mặt Gaussian thật, bề mặt MLP xấp xỉ và biểu đồ phân tán y_true vs y_pred."""
    metrics, y_true_test, y_pred_test = evaluate_predictions(model, test_loader, device)
    
    grid_x1 = np.linspace(config.X1_RANGE[0], config.X1_RANGE[1], 60)
    grid_x2 = np.linspace(config.X2_RANGE[0], config.X2_RANGE[1], 60)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Giá trị hàm thật
    Y_true_grid = gaussian_2d(grid_points, config.MU, config.C).reshape(X1.shape)
    
    # Giá trị dự đoán từ mạng MLP
    model.eval()
    with torch.no_grad():
        tensor_grid = torch.tensor(grid_points, dtype=torch.float32).to(device)
        Y_pred_grid = model(tensor_grid).cpu().numpy().reshape(X1.shape)

    fig = plt.figure(figsize=(16, 5))
    
    # 1. Bề mặt hàm thật
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X1, X2, Y_true_grid, cmap='viridis', alpha=0.85)
    ax1.set_title("Bề mặt hàm Gaussian thật f(x1, x2)")
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_zlabel("y")

    # 2. Bề mặt xấp xỉ bởi MLP
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1, X2, Y_pred_grid, cmap='plasma', alpha=0.85)
    ax2.set_title("Bề mặt xấp xỉ bởi mạng MLP tối ưu")
    ax2.set_xlabel("x1"); ax2.set_ylabel("x2"); ax2.set_zlabel("y_hat")

    # 3. Tương quan y_true và y_pred trên tập Test 100 mẫu
    ax3 = fig.add_subplot(133)
    ax3.scatter(y_true_test, y_pred_test, color='dodgerblue', edgecolor='k', alpha=0.8, label='Mẫu Test (N=100)')
    min_val, max_val = float(y_true_test.min()), float(y_true_test.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường lý tưởng (y = y_hat)')
    ax3.set_title(f"Test Set: MSE={metrics['mse']:.5f} | RMSE={metrics['rmse']:.4f}")
    ax3.set_xlabel("Giá trị thật (y_true)")
    ax3.set_ylabel("Giá trị dự đoán (y_pred)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return metrics

if __name__ == "__main__":
    # 1. Đảm bảo dữ liệu đã tồn tại
    if not os.path.exists("data/train_max.csv"):
        os.makedirs("data", exist_ok=True)
        sample_dataset(config.TEST_SIZE, seed=config.SEED_DATA + 1).to_csv("data/test.csv", index=False)
        sample_dataset(config.VAL_SIZE, seed=config.SEED_DATA + 2).to_csv("data/val.csv", index=False)
        df_tr = sample_dataset(config.TRAIN_MAX_SIZE, seed=config.SEED_DATA + 3)
        df_tr.to_csv("data/train_max.csv", index=False)
        plot_data_exploration(df_tr, "figures/data_distribution_3d.png")

    # 2. Chạy tuần tự 3 nhóm thực nghiệm A, B, C
    df_a = run_experiment_a()
    df_b = run_experiment_b()
    df_c = run_experiment_c()

    # 3. Chọn cấu hình tối ưu dựa trên Validation MSE thấp nhất từ A, B, C
    best_A = int(df_a.loc[df_a["val_mse_mean"].idxmin(), "A_train_size"])
    best_B = int(df_b.loc[df_b["val_mse_mean"].idxmin(), "B_num_neurons"])
    best_C = int(df_c.loc[df_c["val_mse_mean"].idxmin(), "C_num_layers"])
    optimal_arch = [best_B] * best_C

    print("\n=======================================================")
    print(f"CẤU HÌNH TỐI ƯU ĐƯỢC CHỌN TỪ VALIDATION SET:")
    print(f"- Số mẫu huấn luyện tốt nhất (A): {best_A}")
    print(f"- Độ rộng tốt nhất (B): {best_B} nơ-ron/tầng")
    print(f"- Độ sâu tốt nhất (C): {best_C} tầng ẩn -> Kiến trúc: {optimal_arch}")
    print("=======================================================")

    # 4. Huấn luyện mô hình tối ưu và đánh giá cuối cùng trên tập Test (100 mẫu)
    device = get_device()
    cfg = config.BASELINE_CONFIG
    set_seed(cfg["seeds"][0])
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_size=best_A, batch_size=cfg["batch_size"]
    )
    optimal_model = FlexibleMLP(
        input_dim=2, hidden_layers=optimal_arch, 
        output_dim=1, activation=cfg["activation"]
    )
    optimal_model, _, _, best_ep, _ = train_model(
        optimal_model, train_loader, val_loader, epochs=cfg["epochs"],
        lr=cfg["lr"], optimizer_name=cfg["optimizer"],
        checkpoint_path="checkpoints/best_optimal_model.pt", device=device
    )

    test_metrics = visualize_optimal_approximation(optimal_model, test_loader, device)
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST (100 MẪU) [Best Epoch: {best_ep}]:")
    print(f"-> Test MSE : {test_metrics['mse']:.6f}")
    print(f"-> Test MAE : {test_metrics['mae']:.6f}")
    print(f"-> Test RMSE: {test_metrics['rmse']:.6f}")
    print("Đã xuất toàn bộ bảng kết quả tại thư mục logs/ và đồ thị tại thư mục figures/!")


#lưu file excel
    test_metrics = visualize_optimal_approximation(optimal_model, test_loader, device)
    
    # Gọi hàm lưu toàn bộ thực nghiệm sang file .xlsx
    optimal_info = {
        "best_A": best_A,
        "best_B": best_B,
        "best_C": best_C,
        "optimal_arch": optimal_arch,
        "test_metrics": test_metrics
    }
    export_experiments_to_excel(
        cfg=cfg, df_a=df_a, df_b=df_b, df_c=df_c, 
        optimal_info=optimal_info, 
        filepath="logs/MLP_Experimental_Results.xlsx"
    )
    print(f"Hoàn tất! Test MSE: {test_metrics['mse']:.6f} | Test MAE: {test_metrics['mae']:.6f} | Test RMSE: {test_metrics['rmse']:.6f}")