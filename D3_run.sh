#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# # Preprocess tweets
# python3 src/preprocess_olid.py \
#     --file data/olid-training-v1.0.tsv \
#     --train_ids data/train_ids.txt \
#     --val_ids data/val_ids.txt \
#     --split_punctuation \
#     --remove_apostraphes \
#     --remove_hashtags

# # Train LSTM classifier
# python3 src/lstm_train.py \
#     --config configs/D3.json \
#     --train_data data/clean_train_olid.tsv \
#     --val_data data/clean_val_olid.tsv \
#     --model_config_path lstm_saved_configs \
#     --train_tokens_only 

# Run LSTM predictions and generate output
python3 src/lstm_predict.py \
    --config configs/D3.json \
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
    --model_config_path lstm_saved_configs \
    --model_path models/lstm_D3_best_model.pt \
    --val_output_csv outputs/D3/D3_val_preds.csv \

# Evaluation script
python3 src/eval.py \
    --val_output_csv outputs/D3/D3_val_preds.csv \
    --output_path results/D3_scores.out
