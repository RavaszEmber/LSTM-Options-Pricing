import torch
import torch.nn as nn


class LSTMOptionPricer(nn.Module):
    """
    Deep LSTM stack adapted for the transformers training pipeline.

    Constructor accepts `n_features` (will be injected by train.py) and
    other hyperparameters via kwargs. Forward signature matches the
    rest of the codebase: forward(x_enc, x_dec).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 6,
        batch_first: bool = True,
        dropout_last: float = 0.0,
        **kwargs
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Build stacked single-layer LSTMs with BatchNorm + ReLU between layers
        lstms = []
        bns = []
        in_sz = self.n_features

        for _ in range(self.num_layers):
            lstms.append(
                nn.LSTM(
                    input_size=in_sz,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=self.batch_first,
                    bidirectional=False,
                )
            )
            bns.append(nn.BatchNorm1d(self.hidden_size))
            in_sz = self.hidden_size

        self.lstm_layers = nn.ModuleList(lstms)
        self.bn_layers = nn.ModuleList(bns)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_last)
        self.final_fc = nn.Linear(self.hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with stable defaults for LSTMs and linear layers."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if "weight_ih" in name:
                            nn.init.xavier_uniform_(param)
                        elif "weight_hh" in name:
                            nn.init.orthogonal_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)
                            # set forget gate bias to 1
                            H = param.shape[0] // 4
                            param[H:2 * H].fill_(1.0)

    def forward(self, x_enc, x_dec=None):
        """
        Args:
            x_enc: Tensor (batch_size, seq_len, n_features)
            x_dec: ignored (kept for API compatibility)

        Returns:
            Tensor (batch_size, 1)
        """
        y = x_enc

        for lstm, bn in zip(self.lstm_layers, self.bn_layers):
            y, _ = lstm(y)  # (N, T, H)

            # BatchNorm1d expects (N, C) so reshape (N*T, H)
            N, T, H = y.shape
            y = y.reshape(N * T, H)
            y = bn(y)
            y = self.relu(y)
            y = y.reshape(N, T, H)

        # take last timestep
        last = y[:, -1, :]
        last = self.dropout(last)
        out = self.final_fc(last)
        return out


__all__ = ["LSTMOptionPricer"]
