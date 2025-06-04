"""
Naive RNN similar to the one in naiveRNN.py, but with sequence handling and
batch processing.
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, in_layer: int, out_layer: int, hidden_layer: int):
        super().__init__()
        self.in_layer = in_layer
        self.hidden_layer = hidden_layer
        self.out_layer = out_layer

        # Weight matrices (learnable parameters)
        # W_xh: (in_layer, hidden_layer) - maps input to hidden space
        self.W_xh = nn.Parameter(torch.randn(in_layer, hidden_layer))

        # W_hh: (hidden_layer, hidden_layer) - recurrent connections
        self.W_hh = nn.Parameter(torch.randn(hidden_layer, hidden_layer))

        # W_hy: (hidden_layer, out_layer) - maps hidden to output space
        self.W_hy = nn.Parameter(torch.randn(hidden_layer, out_layer))

        # Bias vectors (learnable parameters)
        # bias_h: (hidden_layer,) - shifts hidden state computation
        self.bias_h = nn.Parameter(torch.randn(hidden_layer))

        # bias_o: (out_layer,) - shifts output computation
        self.bias_o = nn.Parameter(torch.randn(out_layer))

    def forward(self, x: torch.Tensor, init_hidden: torch.Tensor | None = None):
        """
        Processes a batch of sequences through the RNN.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, in_layer).
            init_hidden: Initial hidden state tensor of shape (batch_size, hidden_layer).
                         If None, it defaults to a zero tensor.

        Returns:
            A tuple (outputs, last_hidden_state):
            - outputs: Tensor of shape (batch_size, sequence_length, out_layer)
                       containing the output from each time step for each sequence.
            - last_hidden_state: Tensor of shape (batch_size, hidden_layer)
                                 representing the final hidden state for each sequence.
        """
        batch_size, seq_len, _ = x.shape

        if init_hidden is None:
            # initializing hidden state with zeros if not provided.
            # ensuring it's on the same device and dtype as the input.
            h_t_prev = torch.zeros(
                batch_size,
                self.hidden_layer,
            )
        else:
            h_t_prev = init_hidden

        # storing outputs from each time step
        outputs_over_time = []

        # iterating over the sequence length
        for t in range(seq_len):
            # getting all batch inputs at the current time step t
            # x_t shape: (batch_size, in_layer)
            x_t = x[:, t, :]

            # RNN cell computation:
            # h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + bias_h)
            # x_t @ W_xh: (batch_size, in_layer) @ (in_layer, hidden_layer) -> (batch_size, hidden_layer)
            # h_t_prev @ W_hh: (batch_size, hidden_layer) @ (hidden_layer, hidden_layer) -> (batch_size, hidden_layer)
            # bias_h: (hidden_layer,) - needs broadcasting to (batch_size, hidden_layer)
            h_t_current = torch.tanh(
                torch.matmul(x_t, self.W_xh)
                + torch.matmul(h_t_prev, self.W_hh)
                + self.bias_h
            )  # bias_h will be broadcasted

            # Output computation:
            # y_t = h_t_current @ W_hy + bias_o
            # h_t_current @ W_hy: (batch_size, hidden_layer) @ (hidden_layer, out_layer) -> (batch_size, out_layer)
            # bias_o: (out_layer,) - needs broadcasting to (batch_size, out_layer)
            y_t = (
                torch.matmul(h_t_current, self.W_hy) + self.bias_o
            )  # bias_o will be broadcasted

            outputs_over_time.append(y_t)
            h_t_prev = h_t_current  # Update hidden state for next time step

        # stacking the outputs along the sequence dimension
        # Each y_t is (batch_size, out_layer).
        # We want (batch_size, sequence_length, out_layer)
        outputs = torch.stack(outputs_over_time, dim=1)

        # the final h_t_current is the last hidden state
        return outputs, h_t_current


if __name__ == "__main__":
    # parameters for the RNN
    in_features = 5  # Dimensionality of input features (in_layer)
    # Number of features in the hidden state (hidden_layer)
    hidden_features = 10
    out_features = 2  # Dimensionality of output features (out_layer)

    # parameters for the input data
    batch_size = 4
    sequence_length = 8

    # creating RNN instance
    rnn_model = RNN(
        in_layer=in_features, out_layer=out_features, hidden_layer=hidden_features
    )

    # creating dummy input data: (batch_size, sequence_length, in_features)
    dummy_input_sequence = torch.randn(batch_size, sequence_length, in_features)

    # Test Case 1: Forward pass without providing initial hidden state
    # the model will initialize hidden state to zeros.
    print("Test Case 1: Auto-initialized hidden state")
    outputs, last_hidden = rnn_model(dummy_input_sequence)

    print(f"Input shape: {dummy_input_sequence.shape}")
    print(
        f"Outputs shape: {outputs.shape}"
    )  # Expected: (batch_size, sequence_length, out_features)
    print(
        f"Last hidden state shape: {last_hidden.shape}"
    )  # Expected: (batch_size, hidden_features)
    print(
        f"Example output for first item in batch, first time step: {
            outputs[0, 0, :]}"
    )
    print(
        f"Example last hidden state for first item in batch: {
            last_hidden[0, :]}\n"
    )

    # Test Case 2: Forward pass with an explicit initial hidden state (zeros)
    print("Test Case 2: Explicit initial hidden state (zeros)")
    initial_hidden_state = torch.zeros(batch_size, hidden_features)
    outputs_with_init_h, last_hidden_with_init_h = rnn_model(
        dummy_input_sequence, initial_hidden_state
    )

    print(f"Input shape: {dummy_input_sequence.shape}")
    print(f"Outputs shape (with init_h): {outputs_with_init_h.shape}")
    print(
        f"Last hidden state shape (with init_h): {
            last_hidden_with_init_h.shape}"
    )

    # verifying that providing zeros explicitly gives the same output (or very close due to FP arithmetic)
    # this is a good sanity check.
    if torch.allclose(outputs, outputs_with_init_h) and torch.allclose(
        last_hidden, last_hidden_with_init_h
    ):
        print("Results match when providing zero initial hidden state explicitly.\n")
    else:
        print(
            "Mismatch detected when providing zero initial hidden state. Check logic.\n"
        )

    # Test Case 3: Forward pass with a non-zero explicit initial hidden state (random)
    print("Test Case 3: Explicit initial hidden state (random)")
    random_initial_hidden_state = torch.randn(batch_size, hidden_features)
    outputs_with_random_init_h, last_hidden_with_random_init_h = rnn_model(
        dummy_input_sequence, random_initial_hidden_state
    )
    print(f"Input shape: {dummy_input_sequence.shape}")
    print(
        f"Outputs shape (with random init_h): {
            outputs_with_random_init_h.shape}"
    )
    print(
        f"Last hidden state shape (with random init_h): {
            last_hidden_with_random_init_h.shape}"
    )
