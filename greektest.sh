#!/bin/sh

# A script to run a pretrained huggingface model and eval on OffensEval 2020 Greek data
# Note: Set Config parameter in D4.cmd

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home2/davidyi6/ling-573/573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Preprocess tweets
# note: when available, update with greek-specific preprocessing script
python3 src/preprocess_olid.py \
    --file data/offenseval-gr-test-v1.tsv \
    --all_test \
    --split_punctuation \
    --remove_hashtags \
    --language greek

# Run finetuned model predictions on Greek data and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_val_greek.tsv \
    --config configs/finetune_xlmr_large_final_greek/config.json \
    --model_path models/finetune_xlmr_large_final_greek/pytorch_model.bin \
    --val_output_csv outputs/D4/D4_greek_preds_greek-only.csv

# Evaluation script (Greek)
python3 src/eval.py \
    --val_output_csv outputs/D4/D4_greek_preds_greek-only.csv \
    --output_path results/D4_greek_scores_greek-only.out
