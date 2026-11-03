import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[8], output_dim=1, activation="tanh"):
        
        super(FlexibleMLP, self).__init__()
        
        act_dict = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid
        }
        if activation.lower() not in act_dict:
            raise ValueError(f"Hàm kích hoạt {activation} không được hỗ trợ.")
        act_layer = act_dict[activation.lower()]

        layers = []
        in_features = input_dim
        
        # Khởi tạo động các tầng ẩn (nn.Linear + Activation)
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(act_layer())
            in_features = h_dim
            
        # Tầng đầu ra cho bài toán hồi quy (Linear thuần, không dùng activation)
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)