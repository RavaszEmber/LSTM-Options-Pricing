"""Encoder-Decoder Transformer model"""

import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class EncoderDecoderTransformer(nn.Module):
    """Standard Transformer encoder-decoder"""

    def __init__(self, n_features, d_model=64, n_heads=4, 
                 n_encoder_layers=3, n_decoder_layers=3,
                 d_ff=256, dropout=0.1, **kwargs):
        super(EncoderDecoderTransformer, self).__init__()

        self.d_model = d_model
        self.model_type = 'encoder_decoder_transformer'

        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x_enc, x_dec):
        # Embed and encode
        enc_out = self.input_embedding(x_enc)
        enc_out = self.pos_encoder(enc_out)
        enc_out = self.encoder(enc_out)

        # Embed and decode
        dec_out = self.input_embedding(x_dec)
        dec_out = self.pos_encoder(dec_out)
        dec_out = self.decoder(dec_out, enc_out)

        # Project to output
        predictions = self.output_projection(dec_out[:, -1, :])

        return predictions