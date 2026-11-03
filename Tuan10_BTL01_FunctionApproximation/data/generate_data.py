import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    # Khi chạy bằng file .py
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
except NameError:
    # Khi chạy bằng Jupyter Notebook
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)

import config

def gaussian_2d(X, mu, C):
    """Tính giá trị hàm mật độ Gaussian 2 chiều cho mảng X (N, 2)."""
    det_C = np.linalg.det(C)
    inv_C = np.linalg.inv(C)
    diff = X - mu
    # Tính (x - mu)^T * C^(-1) * (x - mu) cho từng mẫu
    exponent = -0.5 * np.sum((diff @ inv_C) * diff, axis=1, keepdims=True)
    coeff = 1.0 / (2.0 * np.pi * np.sqrt(det_C))
    return coeff * np.exp(exponent)

def sample_dataset(n_samples, seed):
    """Lấy mẫu ngẫu nhiên đồng đều trong miền xác định với seed cố định."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(config.X1_RANGE[0], config.X1_RANGE[1], size=(n_samples, 1))
    x2 = rng.uniform(config.X2_RANGE[0], config.X2_RANGE[1], size=(n_samples, 1))
    X = np.hstack([x1, x2])
    y = gaussian_2d(X, config.MU, config.C)
    return pd.DataFrame(np.hstack([X, y]), columns=['x1', 'x2', 'y'])

def plot_data_exploration(df_train, save_path):
    """Vẽ bề mặt hàm thật 3D, scatter dữ liệu và phân bố x1, x2, y."""
    fig = plt.figure(figsize=(16, 5))

    # 1. Bề mặt lý thuyết + 3D Scatter
    ax1 = fig.add_subplot(131, projection='3d')
    grid_x1 = np.linspace(config.X1_RANGE[0], config.X1_RANGE[1], 60)
    grid_x2 = np.linspace(config.X2_RANGE[0], config.X2_RANGE[1], 60)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    Y_true = gaussian_2d(grid_points, config.MU, config.C).reshape(X1.shape)

    ax1.plot_surface(X1, X2, Y_true, cmap='viridis', alpha=0.6)
    ax1.scatter(df_train['x1'][:300], df_train['x2'][:300], df_train['y'][:300], 
                color='red', s=10, label='Mẫu dữ liệu (300 điểm đầu)')
    ax1.set_title("Bề mặt hàm Gaussian 2D & Scatter")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("y")
    ax1.legend()

    # 2. Phân bố x1 và x2
    ax2 = fig.add_subplot(132)
    ax2.hist(df_train['x1'], bins=20, alpha=0.6, label='x1 [-2, 2]')
    ax2.hist(df_train['x2'], bins=20, alpha=0.6, label='x2 [-4, 4]')
    ax2.set_title("Phân bố của x1 và x2")
    ax2.set_xlabel("Giá trị")
    ax2.set_ylabel("Tần suất")
    ax2.legend()

    # 3. Phân bố nhãn y
    ax3 = fig.add_subplot(133)
    ax3.hist(df_train['y'], bins=30, color='green', alpha=0.7)
    ax3.set_title("Phân bố của mục tiêu y = f(x1, x2)")
    ax3.set_xlabel("y")
    ax3.set_ylabel("Tần suất")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # Tách biệt seed cho 3 tập 
    df_test = sample_dataset(config.TEST_SIZE, seed=config.SEED_DATA + 1)
    df_val = sample_dataset(config.VAL_SIZE, seed=config.SEED_DATA + 2)
    df_train = sample_dataset(config.TRAIN_MAX_SIZE, seed=config.SEED_DATA + 3)

    df_test.to_csv("data/test.csv", index=False)
    df_val.to_csv("data/val.csv", index=False)
    df_train.to_csv("data/train_max.csv", index=False)

    plot_data_exploration(df_train, "figures/data_distribution_3d.png")
    print("Đã sinh dữ liệu thành công và xuất biểu đồ tại figures/data_distribution_3d.png!")