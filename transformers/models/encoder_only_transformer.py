"""Encoder Transformer model"""

import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding

class EncoderOnlyTransformer(nn.Module):
    """
    Simplified Transformer model for option pricing with sequence input.
    
    This serves as a baseline before implementing the full Informer architecture.
    
    Reference: Ruiru et al. (2024), Section 2.2, Equation 4, p. 544
    "Attention(Q,K,V) = softmax(QK^T/√dk)V"
    
    Reference: Bańka & Chudziak (2025), Figure 1, p. 1271
    Shows conceptual overview of encoder-decoder architecture
    """
    def __init__(self, n_features, d_model=32, n_heads=4, n_encoder_layers=2, 
                 d_ff=128, dropout=0.1, **kwargs):
        super(EncoderOnlyTransformer, self).__init__()

        self.d_model = d_model
        self.model_type = 'encoder_only_transformer'

        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)

        # Positional embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
    
    def forward(self, x_enc, x_dec=None):
        """
        Args:
            x_enc: Input tensor of shape (batch_size, seq_len, n_features)
            x_dec: Decoder input (ignored, for API compatibility)
        Returns:
            predictions: Tensor of shape (batch_size, 1)
        """
        # Embed encoder input
        x = self.input_embedding(x_enc)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)

        # Pass through encoder
        x = self.encoder(x)  # (batch_size, seq_len, d_model)

        # Use the last timestep for prediction
        x = x[:, -1, :]  # (batch_size, d_model)

        # Project to output
        predictions = self.output_projection(x)  # (batch_size, 1)

        return predictions