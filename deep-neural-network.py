import torch
import torch.nn as nn

"""
x number of inputs.
y number of outputs.
3 dense layers (between inputs and outputs)

for dense layer 1 (let's call it n1):
    input number: x (to match the nn input)
    output number (this is the neuron count): something
    add non-linearity

for dense layer 2 (n2):
    input number: output of n1
    output number: something
    add non-linearity

for the final dense layer (n3):
    input count: output of n2
    output number: y.
    add non-linearity. note that this non-linearity needs to match the
    expected kind of output from the nn. the format of the output depends
    on what's expected.
"""


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.model = nn.Sequential(
            # layer 1
            nn.Linear(in_features=input_size, out_features=100),
            nn.ReLU(),
            # layer 2
            nn.Linear(in_features=100, out_features=10),
            nn.ReLU(),
            # layer 3
            nn.Linear(in_features=10, out_features=output_size),
            # I could've also added the last non-linearity here like:
            # nn.Softmax(dim=-1)
            # but doing this in the forward method is more flexible,
            # as this outputs raw logits that might be needed for
            # different scenarios
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        out = torch.softmax(out, dim=-1)
        return out


if __name__ == "__main__":
    dnn = DeepNeuralNetwork(input_size=4, output_size=5)
    torch.manual_seed(1)
    output = dnn(torch.tensor([5.0, 2.0, 6.0, -7.0]))
    print(output.shape, output)
