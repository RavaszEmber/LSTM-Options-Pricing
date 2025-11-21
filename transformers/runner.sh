#!/bin/bash

# Encoder Only
./batch_train.sh configs/encoder_only_transformer.yaml 8 ./results/encoder_only/batch

# Encoder-Decoder
./batch_train.sh configs/encoder_decoder_transformer.yaml 8 ./results/encoder_decoder/batch

# Informer
./batch_train.sh configs/informer.yaml 8 ./results/informer/batch

# LSTM32+TX
./batch_train.sh configs/lstm32_transformer.yaml 8 ./results/lstm32_tx/batch