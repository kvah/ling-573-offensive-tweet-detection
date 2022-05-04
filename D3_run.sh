#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch --force-reinstall

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/train_ids.txt \
    --val_ids data/val_ids.txt \
    --split_punctuation True \
    --remove_apostraphes True \
    --remove_hashtags True

# Run LSTM classifier
python3 src/run_lstm.py \
    --config configs/D3.json \
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
