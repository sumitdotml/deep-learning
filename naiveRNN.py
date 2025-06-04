import torch
import torch.nn as nn

"""
Naive RNN Implementation

This implements the fundamental RNN equations from scratch:
    h_t = tanh(h_{t-1} @ W_hh + x_t @ W_xh + bias_h)
    y_t = h_t @ W_hy + bias_y

Where:
    h_t = hidden state at time t
    x_t = input at time t  
    y_t = output at time t
    W_hh = hidden-to-hidden weight matrix
    W_xh = input-to-hidden weight matrix
    W_hy = hidden-to-output weight matrix
"""


class naiveRNN(nn.Module):
    def __init__(self, no_of_rnn_cells: int, input_dim: int, output_dim: int):
        """
        Initialize the RNN with learnable parameters.

        Args:
            no_of_rnn_cells: Size of hidden state (number of memory units)
            input_dim: Dimensionality of input vectors
            output_dim: Dimensionality of output vectors

        The RNN has 3 weight matrices and 2 bias vectors:
            - W_xh: transforms input to hidden space
            - W_hh: transforms previous hidden state to current hidden state
            - W_hy: transforms hidden state to output space
            - bias_hidden: bias for hidden state computation
            - bias_output: bias for output computation
        """
        super().__init__()
        self.no_of_rnn_cells = no_of_rnn_cells

        # Weight matrices (learnable parameters)
        # W_xh: (input_dim, no_of_rnn_cells) - maps input to hidden space
        self.W_xh = nn.Parameter(torch.randn(input_dim, no_of_rnn_cells))

        # W_hh: (no_of_rnn_cells, no_of_rnn_cells) - recurrent connections
        self.W_hh = nn.Parameter(torch.randn(no_of_rnn_cells, no_of_rnn_cells))

        # W_hy: (no_of_rnn_cells, output_dim) - maps hidden to output space
        self.W_hy = nn.Parameter(torch.randn(no_of_rnn_cells, output_dim))

        # Bias vectors (learnable parameters)
        # bias_hidden: (no_of_rnn_cells,) - shifts hidden state computation
        self.bias_hidden = nn.Parameter(torch.randn(no_of_rnn_cells))

        # bias_output: (output_dim,) - shifts output computation
        self.bias_output = nn.Parameter(torch.randn(output_dim))

    def forward(
        self, x: torch.Tensor, prev_hidden_state: torch.Tensor | None = None
    ) -> list[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN for a single timestep.

        Args:
            x: Input tensor of shape (input_dim,)
            prev_hidden_state: Previous hidden state of shape (no_of_rnn_cells,)
                              If None, initializes to zeros (start of sequence)

        Returns:
            tuple: (output, updated_hidden_state)
                - output: Output tensor of shape (output_dim,)
                - updated_hidden_state: New hidden state of shape (no_of_rnn_cells,)

        Mathematical Flow:
            1. Combine previous memory (h_{t-1}) and current input (x_t)
            2. Apply non-linearity (tanh) to get new hidden state (h_t)
            3. Transform hidden state to output space (y_t)
        """

        # Initializing hidden state to zeros if this is the start of a sequence
        if prev_hidden_state is None:
            prev_hidden_state = torch.zeros(self.no_of_rnn_cells)

        # Step 1: Computing new hidden state
        # Formula: h_t = tanh(h_{t-1} @ W_hh + x_t @ W_xh + bias_hidden)
        #
        # Dimension analysis for the computation:
        # - prev_hidden_state: (no_of_rnn_cells,) @ W_hh: (no_of_rnn_cells, no_of_rnn_cells)
        #   = (no_of_rnn_cells,) [memory contribution]
        # - x: (input_dim,) @ W_xh: (input_dim, no_of_rnn_cells)
        #   = (no_of_rnn_cells,) [input contribution]
        # - bias_hidden: (no_of_rnn_cells,) [learnable shift]
        # - Final result: (no_of_rnn_cells,) [new hidden state]
        updated_hidden_state = torch.tanh(
            prev_hidden_state @ self.W_hh + x @ self.W_xh + self.bias_hidden
        )

        # Step 2: Computing output from new hidden state
        # Formula: y_t = h_t @ W_hy + bias_output
        #
        # Dimension analysis for the computation:
        # - updated_hidden_state: (no_of_rnn_cells,) @ W_hy: (no_of_rnn_cells, output_dim)
        #   = (output_dim,) [raw output]
        # - bias_output: (output_dim,) [learnable shift]
        # - Final result: (output_dim,) [final output]
        out = updated_hidden_state @ self.W_hy + self.bias_output

        return out, updated_hidden_state


if __name__ == "__main__":

    no_of_rnn_cells = 3
    input_dim = 2
    output_dim = 1

    rnn = naiveRNN(no_of_rnn_cells, input_dim, output_dim)

    x = torch.randn(input_dim)  # Random input: (2,)
    prev_hidden_state = torch.randn(no_of_rnn_cells)  # Random initial state: (3,)

    out, updated_hidden_state = rnn(x, prev_hidden_state)

    print("Output shape:", out.shape)  # Should be: torch.Size([1])
    print("Output:", out)
    print(
        "Updated hidden state shape:", updated_hidden_state.shape
    )  # Should be: torch.Size([3])
    print("Updated hidden state:", updated_hidden_state)

    # For sequential processing, I could do:
    # h = torch.zeros(no_of_rnn_cells)  # Initial hidden state
    # for x_t in sequence:
    #     out_t, h = rnn(x_t, h)  # h carries memory forward
    #     # Process out_t...
    # will do this in my next marathon run
