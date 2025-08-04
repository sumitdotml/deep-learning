import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Forward pass for training
        src: [batch_size, src_seq_len] - source sequence (e.g., German sentence)
        tgt: [batch_size, tgt_seq_len] - target sequence (e.g., English sentence)
        Returns:
            predictions: [batch_size, tgt_seq_len-1, tgt_vocab_size] - predicted target sequence logits
            expected_output: [batch_size, tgt_seq_len-1] - expected target sequence (token IDs)
        """
        # STEP 1: ENCODE THE SOURCE SEQUENCE
        # Pass the entire source sequence through encoder to get context
        # The encoder will return hidden_state and cell_state that capture the meaning
        # of the entire source sequence
        # TODO: call self.encoder(src) and capture hidden_state, cell_state
        hidden_state, cell_state = self.encoder(src)
        hidden_state, cell_state = hidden_state.to(self.device), cell_state.to(
            self.device
        )

        # STEP 2: PREPARE FOR DECODING
        # During training, we use "teacher forcing" - we feed the correct previous
        # target token to predict the next token (rather than using our own predictions)
        #
        # Key insight: if target is [<START>, "Hola", "mundo", <END>]
        # - Decoder input should be: [<START>, "Hola", "mundo"] (all but last)
        # - Expected output should be: ["Hola", "mundo", <END>] (all but first)
        # This is called "shifted teacher forcing"
        # TODO: create decoder_input by removing the last token from tgt
        decoder_input = tgt[:, :-1].to(self.device)

        # TODO: create expected_output by removing the first token from tgt
        expected_output = tgt[:, 1:].to(self.device)

        # STEP 3: DECODE STEP BY STEP
        # The decoder processes one token at a time, but we can be clever during training
        # We'll loop through each position in the target sequence and:
        # 1. Feed the decoder the current input token + previous hidden/cell_state
        # 2. Get projection + updated hidden/cell_state
        # 3. Store the projection for this position
        # 4. Use updated hidden, cell_state for next position
        # TODO: create empty tensor to store all decoder projections
        decoder_projections = []

        # TODO: loop through each position in decoder_input sequence
        # TODO: for each position, call decoder with current input + hidden + cell_state
        # TODO: collect all decoder projections into a single tensor
        for i in range(decoder_input.shape[1]):

            # teacher forcing: using ground truth token, not the prediction
            current_input = decoder_input[:, i]
            projection, decoder_hidden, decoder_cell_state = self.decoder(
                current_input, hidden_state, cell_state
            )
            decoder_projections.append(projection)
            hidden_state, cell_state = decoder_hidden, decoder_cell_state

        # STEP 4: RETURN RESULTS
        # Return all decoder projections so we can compute loss against expected_output
        # Shape should be [batch_size, tgt_seq_len-1, tgt_vocab_size]
        # (tgt_seq_len-1 because we removed either first or last token)
        # TODO: return the collected decoder projections

        predictions = torch.stack(decoder_projections, dim=1)

        return predictions, expected_output

    def generate(
        self, src: torch.Tensor, start_token: int, end_token: int, max_length: int = 50
    ) -> torch.Tensor:
        """
        Inference method - generates target sequence autoregressively
        src: [batch_size, src_seq_len] - source sequence
        start_token: token ID for <START>
        end_token: token ID for <END>
        max_length: maximum sequence length to generate
        """

        # STEP 1: ENCODE SOURCE (same as training)
        # TODO: encode src to get initial hidden_state, cell_state
        hidden_state, cell_state = self.encoder(src)
        hidden_state, cell_state = hidden_state.to(self.device), cell_state.to(
            self.device
        )

        # STEP 2: INITIALIZE GENERATION
        # Start with <START> token for each example in batch
        # Shape should be [batch_size] containing start_token for each example
        # TODO: create initial input tensor with start_token
        current_input = torch.full(
            size=(src.shape[0],), fill_value=start_token, device=self.device
        )

        # Store generated tokens for each example in batch
        # TODO: create empty list to store generated sequences
        generated_sequences = []

        # STEP 3: AUTOREGRESSIVE GENERATION LOOP
        # Unlike training, we don't know the target length ahead of time
        # We generate one token at a time until:
        # 1. We predict <END> token, OR
        # 2. We reach max_length
        # TODO: loop for max_length iterations
        #   TODO: pass current_input + hidden + cell_state to decoder
        #   TODO: get prediction logits + updated states
        #   TODO: convert logits to predicted token (argmax or sampling)
        #   TODO: store predicted token
        #   TODO: use predicted token as next input (no teacher forcing!)
        #   TODO: update hidden, cell_state for next iteration
        #   TODO: check if all examples in batch predicted <END> token (optional early stopping)
        for _ in range(max_length):
            projection, decoder_hidden, decoder_cell_state = self.decoder(
                current_input, hidden_state, cell_state
            )
            predicted_token = torch.argmax(projection, dim=-1)
            generated_sequences.append(predicted_token)

            # if all examples in the batch predicted the end token, stop generating
            if torch.all(predicted_token == end_token):
                break

            hidden_state, cell_state = decoder_hidden, decoder_cell_state
            current_input = predicted_token

        # STEP 4: FORMAT OUTPUT
        # Convert list of generated tokens into proper tensor format
        # TODO: stack/pad generated sequences into tensor
        # TODO: return generated sequences
        predictions = torch.stack(generated_sequences, dim=1)
        return predictions
