import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(train_csv="data/train_max.csv", 
                       val_csv="data/val.csv", 
                       test_csv="data/test.csv", 
                       train_size=1000, 
                       batch_size=32):
   
    df_train = pd.read_csv(train_csv).iloc[:train_size]
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    def df_to_dataset(df):
        X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
        y = torch.tensor(df[['y']].values, dtype=torch.float32)
        return TensorDataset(X, y)

    train_dataset = df_to_dataset(df_train)
    val_dataset = df_to_dataset(df_val)
    test_dataset = df_to_dataset(df_test)

    # Bật shuffle=True cho tập train, tắt cho tập val và test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, val_loader, test_loader