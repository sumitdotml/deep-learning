from typing import Optional

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.randn(hidden_size * 4, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size * 4, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size * 4))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size * 4))

    def forward(
        self, input: torch.Tensor, hidden: Optional[list[torch.Tensor]] = None
    ) -> list[torch.Tensor]:
        batch_size = input.shape[0]
        if hidden is not None:
            h_prev, c_prev = hidden
        else:
            h_prev = torch.zeros(batch_size, self.hidden_size)
            c_prev = torch.zeros(batch_size, self.hidden_size)

        gi = input @ self.W_ih.T + self.b_ih
        gh = h_prev @ self.W_hh.T + self.b_hh

        i_input_gate, i_forget_gate, i_candidate, i_output_gate = gi.chunk(
            4, 1)
        h_input_gate, h_forget_gate, h_candidate, h_output_gate = gh.chunk(
            4, 1)

        forget_gate = torch.sigmoid(i_forget_gate + h_forget_gate)
        input_gate = torch.sigmoid(i_input_gate + h_input_gate)
        candidate_values = torch.tanh(i_candidate + h_candidate)
        output_gate = torch.sigmoid(i_output_gate + h_output_gate)

        c_new = c_prev * forget_gate + input_gate * candidate_values
        h_new = torch.tanh(c_new) * output_gate

        return [h_new, c_new]
