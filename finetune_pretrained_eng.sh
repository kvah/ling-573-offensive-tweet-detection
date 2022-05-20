#!/bin/sh

# A script to finetune a pretrained huggingface model and eval on OLID data
# Note: Set Config parameter in D4.cmd

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/eng_train_ids.txt \
    --val_ids data/eng_val_ids.txt \
    --split_punctuation \
    --remove_apostraphes \
    --remove_hashtags

# Finetune pretrained model on training data
python3 src/finetune_pretrained.py \
    --train_data data/clean_train_english.tsv \
    --val_data data/clean_val_english.tsv \
    --config configs/${1}.json

# Run finetuned model predictions and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_val_english.tsv \
    --config configs/${1}.json \
    --model_path models/${1} \
    --val_output_csv outputs/D4/D4_english_preds.csv

# Evaluation script
python3 src/eval.py \
    --val_output_csv outputs/D4/D4_english_preds.csv \
    --output_path results/D4_english_scores.out
