#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Finetune pretrained model on training data
python3 src/finetune_pretrained.py \
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
    --config configs/finetune_xlmr_2.json

# Run finetuned model predictions and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_val_olid.tsv \
    --config configs/finetune_xlmr.json \
    --model_path models/finetune_xlmr_2 \
    --val_output_csv outputs/D4/D4_val_preds.csv

# Evaluation script
python3 src/eval.py \
    --val_output_csv outputs/D4/D4_val_preds.csv \
    --output_path results/D4_scores.out
