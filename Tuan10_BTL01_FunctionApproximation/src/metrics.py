import numpy as np
import torch

def evaluate_predictions(model, data_loader, device):
    """Dự đoán trên tập dữ liệu và trả về bộ 3 chỉ số MSE, MAE, RMSE."""
    model.eval()
    preds_list, targets_list = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()
            preds_list.append(outputs)
            targets_list.append(y_batch.numpy())
            
    y_pred = np.vstack(preds_list)
    y_true = np.vstack(targets_list)
    
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    
    return {"mse": mse, "mae": mae, "rmse": rmse}, y_true, y_pred