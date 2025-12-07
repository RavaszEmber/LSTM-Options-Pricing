#!/bin/bash

# LSTM 
./batch_train.sh configs/lstm_option_pricer.yaml 8 ./results/lstm_option_pricer

# Encoder Only
./batch_train.sh configs/encoder_only.yaml 8 ./results/encoder_only

# Encoder-Decoder
./batch_train.sh configs/encoder_decoder.yaml 8 ./results/encoder_decoder

# Informer
./batch_train.sh configs/informer.yaml 8 ./results/informer

# LSTM32+TX
./batch_train.sh configs/lstm32_tx.yaml 8 ./results/lstm32_tx

# Pimental MLP
./batch_train.sh configs/pimental_mlp.yaml 8 ./results/pimental_mlp
./batch_train.sh configs/pimental_mlp_gg.yaml 8 ./results/pimental_mlp_gg