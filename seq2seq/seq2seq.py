import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
