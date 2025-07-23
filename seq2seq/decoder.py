import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=embedding_dim
        )

        self.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.proj = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, cell_state: torch.Tensor
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

        # [batch_size] -> [1, batch_size]
        input = input.unsqueeze(0)

        # [1, batch_size] ->  [1,  batch_size, embedding_dim]
        embedded_input = self.embedding(input)

        # output: [1, batch_size, hidden_dim]
        output, (hidden, cell_state) = self.LSTM(
            embedded_input, (hidden, cell_state))

        # [1,  batch_size, hidden_dim] -> [batch_size, hidden_dim]
        projection = self.proj(output.squeeze(0))

        return projection, hidden, cell_state
