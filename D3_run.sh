#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./env

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/train_ids.txt \
    --dev_ids data/val_ids.txt

# Run LSTM classifier
python3 src/run_lstm.py \
    --train_data data/pp_train_olid-training-v1.0.tsv \
    --val_data data/pp_dev_olid-training-v1.0.tsv \
    --num_epochs 20