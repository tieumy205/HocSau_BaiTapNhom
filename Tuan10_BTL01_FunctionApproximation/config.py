import numpy as np

# 1. Thông số bài toán Gaussian 2D
MU = np.array([0.0, 0.0])
C = np.array([[0.3, -0.5], 
              [-0.5, 2.0]])

# Miền giá trị lấy mẫu
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

# 2. Quy định kích thước các tập dữ liệu và Seed cố định
SEED_DATA = 42
TEST_SIZE = 100        
VAL_SIZE = 400         
TRAIN_MAX_SIZE = 2000  

# 3. Cấu hình Baseline mặc định
BASELINE_CONFIG = {
    "train_size": 1000,
    "hidden_layers": [8],   
    "activation": "tanh",   
    "optimizer": "adam",
    "lr": 0.01,
    "batch_size": 32,
    "epochs": 500,
    "seeds": [42, 123, 999]
}