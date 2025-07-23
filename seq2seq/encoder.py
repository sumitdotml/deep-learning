import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        # shape: [input_size, embedding_dim]
        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_dim
        )

        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
        # here, x has to be [batch_size, seq_len] to begin with

        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        x_embedded = self.dropout(self.embedding(x))

        # the torch LSTM outputs out, (hidden, cell). we don't need the
        # `out` for Encoder since this is seq2seq
        _, (hidden_state, cell_state) = self.LSTM(x_embedded)

        return hidden_state, cell_state


if __name__ == "__main__":
    encoder = Encoder(
        input_size=10, embedding_dim=5, hidden_size=3, num_layers=10, dropout=0.1
    )
    x = torch.randint(0, 10, (2, 5))
    hidden_state, cell_state = encoder(x)
    print(f"hidden_state.shape: {hidden_state.shape}")
    print(f"cell_state.shape: {cell_state.shape}")
