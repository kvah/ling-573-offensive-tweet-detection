#!/bin/sh

# A script to finetune a pretrained huggingface model and eval on OffensEval 2020 Greek data
# Note: Set Config parameter in D4.cmd

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ./573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Preprocess Greek tweets
python3 src/preprocess_olid.py \
    --file data/offenseval-gr-training-v1.tsv \
    --train_ids data/gr_train_ids.txt \
    --val_ids data/gr_val_ids.txt \
    --split_punctuation \
    --remove_hashtags \
    --language greek
    
# Preprocess English tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/eng_train_ids.txt \
    --val_ids data/eng_val_ids.txt \
    --split_punctuation \
    --remove_apostraphes \
    --remove_hashtags

# Finetune pretrained model on training data
python3 src/finetune_pretrained.py \
    --train_data data/clean_train_greek.tsv data/clean_train_english.tsv \
    --val_data data/clean_val_greek.tsv data/clean_val_english.tsv \
    --config configs/${1}.json

# Run finetuned model predictions on Greek data and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_val_greek.tsv \
    --config configs/${1}.json \
    --model_path models/${1} \
    --val_output_csv outputs/D4/D4_greek_preds_hybrid.csv

# Evaluation script (Greek)
python3 src/eval.py \
    --val_output_csv outputs/D4/D4_greek_preds_hybrid.csv \
    --output_path results/D4_greek_scores_hybrid.out

# Run finetuned model predictions on English data and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_val_english.tsv \
    --config configs/${1}.json \
    --model_path models/${1} \
    --val_output_csv outputs/D4/D4_english_preds_hybrid.csv

# Evaluation script (English)
python3 src/eval.py \
    --val_output_csv outputs/D4/D4_english_preds_hybrid.csv \
    --output_path results/D4_english_scores_hybrid.out
    