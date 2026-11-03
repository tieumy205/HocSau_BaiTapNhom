Bài Tập Lớn: Xấp Xỉ Hàm Mật Độ Gaussian 2 Chiều Bằng Mạng MLP (PyTorch)

Dự án triển khai mạng Perceptron đa tầng (MLP) bằng PyTorch để xấp xỉ hàm mật độ xác suất Gaussian 2 chiều và thực hiện 3 nhóm thực nghiệm có kiểm soát (Quy mô dữ liệu A, Độ rộng mạng B, Độ sâu mạng C).


1. Cấu trúc thư mục
- `data/`
    |--`generate_data.py`: Chứa hàm tính toán giải tích f(x) với ma trận C và kỳ vọng. File này sinh và xuất các file .csv độc lập, cố định seed ngay từ đầu để tránh rò rỉ dữ liệu (data leakage)
    |--`test.csv`: chứa khoảng 100 mẫu kiểm thử
    |--`val.csv`: chứa tập kiểm định riêng biệt khoảng 400 mẫu dùng để tính validation loss ở mỗi epoch và chọn checkpoint tốt nhất
    |--`train_max.csv`: tập huấn luyện tối đa 2000 mẫu
- `models/mlp.py`: Định nghĩa kiến trúc mạng `FlexibleMLP` kế thừa từ `torch.nn.Module`.
- `src/`: 
    |--`dataset.py` (đọc dữ liệu từ csv, chuyển đổi dữ liệu sang TensorDataset & DataLoader)
    |--`trainer.py` (vòng lặp huấn luyện, tính loss và checkpoiting)
    |--`metrics.py` (tính toán các chỉ số MSE, MAE, RMSE)
    |--`utils.py` (cài đặt cố định random seed) 
    |--`excel_logger.py` (xuất báo cáo Excel).
- `experiments/`: Chứa các script chạy tự động thực nghiệm Nhóm A, Nhóm B, Nhóm C qua 3 random seeds (42, 123, 999).
- `checkpoints/`: Lưu trữ trọng số `.pt` của các mô hình có validation loss thấp nhất.
- `logs/`: Lưu kết quả thực nghiệm dưới dạng `.csv` và file excel `MLP_Experimental_Results.xlsx`.
- `figures/`: Lưu toàn bộ biểu đồ 3D và đồ thị phân tích sai số .
- `config.py`: File cấu hình lưu toàn bộ siêu tham số chuẩn (baseline: lr, batch size, epochs, optimizer, hidden layers)
- `main.py`: Điểm kích hoạt chính: chạy toàn bộ pipline
- `requirements.txt`: Danh sách các thư viện cần cài đặt
- `README.md`: Tài liệu mô tả kiến trúc thư mục, hướng dẫn cài đặt và các lệnh chạy thực nghiệm


2. Cài đặt môi trường
Yêu cầu Python 3.9+ và cài đặt các thư viện qua lệnh:
```bash
pip install -r requirements.txt


    python main.py

3. Hướng dẫn chạy
- Sinh dữ liệu độc lập và vẽ đồ thị khảo sát 3D
    python data/generate_data.py
-Chạy toàn bộ thực nghiệm A, B, C, huấn luyện mô hình tối ưu và xuất đồ thị
    python main.py