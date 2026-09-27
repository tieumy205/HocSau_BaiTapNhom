# src/mlp/model.py

import torch
import torch.nn as nn


class HousePriceMLP(nn.Module):
    """
    MLP dùng cho bài toán House Prices.

    Input:
        Vector feature sau:
        - preprocessing
        - one-hot encoding
        - feature selection

    Output:
        log1p(SalePrice)
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(256, 128, 64),
        dropout=0.2
    ):
        super().__init__()

        layers = []

        in_features = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    in_features,
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            in_features = hidden_dim

        # Output chỉ có 1 giá trị
        layers.append(
            nn.Linear(
                in_features,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        output = self.network(x)

        # từ:
        # (batch_size, 1)
        #
        # thành:
        # (batch_size,)
        return output.squeeze(1)