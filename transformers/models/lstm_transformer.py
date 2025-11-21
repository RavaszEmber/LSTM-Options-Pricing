"""LSTM + Transformer hybrid model - LSTM32_TX only"""

import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class LSTM32_TX(nn.Module):
    """
    LSTM (32 units, 1 layer) + Transformer (4 blocks) hybrid model.
    
    Best performing model from Ruiru et al. (2024):
    - 95.9% accuracy on CF Industries
    - 301,554 parameters
    - Fastest training time among LSTM+TX variants
    """

    def __init__(self, n_features, lstm_units=32, d_model=64, n_heads=4,
                 n_transformer_blocks=4, d_ff=256, dropout=0.1, **kwargs):
        super(LSTM32_TX, self).__init__()

        self.n_features = n_features
        self.lstm_units = lstm_units
        self.d_model = d_model
        self.model_type = 'lstm32_tx'

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True
        )

        # Project LSTM output to transformer dimension
        self.lstm_projection = nn.Linear(lstm_units, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_transformer_blocks
        )

        # Decoder embedding
        self.decoder_embedding = nn.Linear(n_features, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_transformer_blocks
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x_enc, x_dec):
        """Forward pass"""
        # LSTM processing
        lstm_out, _ = self.lstm(x_enc)
        lstm_projected = self.lstm_projection(lstm_out)
        lstm_encoded = self.pos_encoder(lstm_projected)

        # Transformer encoder
        encoder_out = self.transformer_encoder(lstm_encoded)

        # Decoder
        dec_embedded = self.decoder_embedding(x_dec)
        dec_embedded = self.pos_encoder(dec_embedded)
        decoder_out = self.transformer_decoder(dec_embedded, encoder_out)

        # Output
        predictions = self.output_projection(decoder_out[:, -1, :])

        return predictions

    def get_num_params(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)