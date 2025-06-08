import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

"""
Steps:

1. image preprocessing:
    - resize the image to 224x224
    - convert the image to a tensor
    - normalize the image

2. conv layer:
    - it takes in the input image and applies a convolution operation to it
    - the convolution operation is a linear operation that takes in a kernel and a
    - the kernel is a small matrix of weights that is used to convolve over the input image
    - the output of the convolution operation is a feature map
    - the feature map is a matrix of the same size as the input image, but with a different number of channels
    - the number of channels in the feature map is equal to the number of filters in the kernel
    - the number of filters in the kernel is equal to the number of channels in the input image

3. relu:
    - it takes in the feature map and applies a non-linearity to it
    - the non-linearity is a function that is used to introduce non-linearity into the model

4. fully connected layer (just the usual nn) that computes class scores
    - (how each category/class scores)
    - it takes in the feature map and applies a linear operation to it
    - the output of the linear operation is a vector of scores
    - the scores are the scores for each class
    - the scores are then used to compute the loss
    - the loss is then used to update the weights of the model

"""


# 1. preprocess the image
IMG_PATH = pathlib.Path("./assets/kumiko-shimizu-unsplash.jpg")


def preprocess(img_path: pathlib.Path):
    img = Image.open(img_path)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )(img)


x = preprocess(IMG_PATH).unsqueeze(0)  # added batch dim to make it (1, 3, 224, 224)


class naiveCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

        # didn't know this, but LazyLinear seem to automatically infer input size on first forward pass? nice
        self.fc = nn.LazyLinear(out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flattening the tensor for the fully connected layer
        x = x.view(
            x.size(0), -1
        )  # here I'm keeping the batch dimension, flattening the rest
        return self.fc(x)


if __name__ == "__main__":
    print(f"Input: {x.shape}")
    model = naiveCNN(in_channels=3, out_channels=10, kernel_size=3)
    print(f"\nModel:\n{model}")
    print(f"\nOutput shape: {model(x).shape}")
    print(f"Output:\n{model(x)}")
