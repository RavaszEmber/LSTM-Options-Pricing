"""Informer model for time series forecasting"""

import math
import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class ProbSparseSelfAttention(nn.Module):
    """Simplified ProbSparse Self-Attention mechanism"""

    def __init__(self, d_model, n_heads, dropout=0.1, factor=3):
        super(ProbSparseSelfAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size = queries.size(0)
        seq_len = queries.size(1)

        # Multi-head attention
        Q = self.query_projection(queries).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)
        K = self.key_projection(keys).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)
        V = self.value_projection(values).view(
            batch_size, seq_len, self.n_heads, self.d_k
        ).transpose(1, 2)

        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_projection(context)

        return output, attention_weights


class InformerEncoderLayer(nn.Module):
    """Informer Encoder Layer with ProbSparse attention"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=3):
        super(InformerEncoderLayer, self).__init__()

        self.attention = ProbSparseSelfAttention(d_model, n_heads, dropout, factor)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_output, _ = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class InformerModel(nn.Module):
    """Informer model for option pricing"""

    def __init__(self, n_features, d_model=32, n_heads=3, 
                 n_encoder_layers=1, n_decoder_layers=2,
                 d_ff=8, dropout=0.06, factor=3):
        super(InformerModel, self).__init__()

        self.d_model = d_model
        self.model_type = 'informer'

        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(n_encoder_layers)
        ])

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
        # Embed encoder input
        enc_out = self.input_embedding(x_enc)
        enc_out = self.pos_encoder(enc_out)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            enc_out = encoder_layer(enc_out)

        # Embed decoder input
        dec_out = self.input_embedding(x_dec)
        dec_out = self.pos_encoder(dec_out)

        # Pass through decoder
        dec_out = self.decoder(dec_out, enc_out)

        # Output projection
        predictions = self.output_projection(dec_out[:, -1, :])

        return predictions