
import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(
            self,
            array_size,
            input_size,
            cfg,
            device=None
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.array_size = array_size
        self.input_pos_embedding = torch.nn.Embedding(num_embeddings=array_size, embedding_dim=input_size).to(self.device)
        self.input_size = input_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers).to(self.device)

    def positions(self, src_items):
        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        result =  self.input_pos_embedding(pos_encoder)
        return result

    def output_len(self):
        return self.array_size * self.input_size

    def forward(self, src_items):
        src_items = src_items + self.positions(src_items)
        src_items = self.encoder(src_items)
        return src_items
