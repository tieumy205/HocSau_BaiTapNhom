import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=500, lr=0.01, 
                optimizer_name="adam", checkpoint_path="checkpoints/best_model.pt", 
                device="cpu"):
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Chỉ hỗ trợ 'adam' hoặc 'sgd'.")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # --- 1. Pha Huấn luyện (Training) ---
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # --- 2. Pha Kiểm định (Validation) ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val)
                running_val_loss += val_loss.item() * X_val.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        
        # --- 3. Lưu Checkpoint tốt nhất theo Validation Loss ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            
    training_time = time.time() - start_time
    
    # Nạp lại trọng số tốt nhất trước khi trả về mô hình
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    return model, history, best_val_loss, best_epoch, training_time