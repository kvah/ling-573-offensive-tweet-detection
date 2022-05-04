#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./env

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --split_punctuation True \
    --remove_apostraphes True \
    --remove_hashtags True \
    --split_emojis True

# Run LSTM classifier
python3 src/run_lstm.py \
    --config configs/D3.json \
    --train_data data/pp_train_olid-training-v1.0.tsv \
    --val_data data/pp_dev_olid-training-v1.0.tsv \
