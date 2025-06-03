import torch
import torch.nn as nn

"""
3 things important for each neuron:
    - dot product between inputs & their weights
    - add all this result and also add the bias to it
    (bias here is also an input that has its own weight,
    so the added bias is also a result of dot product)
    - apply non-linearity to the added result (sigmoid, relu, etc.)

This gives us y_hat.
"""


class DenseLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(input_size, output_size))

        # shape (out,) – broadcasts to (B, out)
        self.bias = nn.Parameter(torch.randn(1, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # dot product between inputs and weights (and adding bias)
        z = x @ self.weights + self.bias

        # applying non-linearity, e.g., sigmoid
        y_hat = torch.sigmoid(z)
        return y_hat


if __name__ == "__main__":
    input_size = 3
    output_size = 2
    layer = DenseLayer(input_size, output_size)

    x = torch.randn(input_size)

    y_hat = layer(x)

    print("Output shape:", y_hat.shape)
    print("Output:", y_hat)
