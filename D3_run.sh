#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/train_ids.txt \
    --val_ids data/val_ids.txt \
    --split_punctuation True \
    --remove_apostraphes True \
    --remove_hashtags True

# Train LSTM classifier
python3 src/lstm_train.py \
    --config configs/D3.json \
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
    --model_config_path lstm_saved_configs \

# Run LSTM predictions and generate output
python3 src/lstm_predict.py \
    --config configs/D3.json\
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
    --model_config_path lstm_saved_configs \
    --model_path models/lstm_D3_best_model.pt \
    --fig_path outputs/D3_F1_curve.png \
    --val_output_csv outputs/D3_val_preds.csv

# Evaluation script
python3 src/eval.py \
    --val_output_csv outputs/D3_val_preds.csv \
    --output_path results/D3_scores.out
