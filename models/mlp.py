"""
R. Pimentel et al. Table 3 defines the MLP structures

It follows that it is not able to capture information embedded in the time-series structure as
the LSTM model. The implementation and training of the MLP follows the same
procedure as the LSTM. We also conduct a hyperparameter search for the MLP to
derive the hyperparameter values shown in Table 3.

Uses same input as B.S., so
    S_0 - Current price of the underlying asset
    K - Exercise (strike) price of the option
    T - Time to maturity
    r - Risk-free interest rate
    sigma - Volatility of the underlying asset

"""

import torch
import torch.nn as nn

import logging


class PimentelMLP(nn.Module):
    def __init__(
        self,
        input_feature_size=5,
        output_size=1,
        number_layers=4,
        units_per_layer=64,
        device="cpu",
        momentum=0.1,
    ):
        super().__init__()
        self._input_feature_size = input_feature_size
        self._output_size = output_size
        self._number_layers = number_layers
        self._units_per_layer = units_per_layer
        self._device = device
        self._momentum = momentum

        logging.info(
            f"Pimentel MLP initialized. {self._input_feature_size=} {self._output_size=} {self._number_layers=} {self._units_per_layer=} {self._device=} {self._momentum=}"
        )

        layers = self.generate_layers()
        self._network = nn.Sequential(*layers)

    def generate_layers(self):
        layers = []
        for layer in range(self._number_layers):
            in_features = self._units_per_layer
            out_features = self._units_per_layer

            is_first_layer = layer == 0
            is_last_layer = layer == self._number_layers - 1

            if is_first_layer:
                in_features = self._input_feature_size

            if is_last_layer:
                out_features = self._output_size

            layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    device=self._device,
                )
            )
            if not is_last_layer:
                layers.append(
                    nn.BatchNorm1d(
                        num_features=out_features,
                        momentum=self._momentum,
                        device=self._device,
                    )
                )
                layers.append(nn.ReLU())
        return layers

    def forward(self, X):
        return self._network(X)
