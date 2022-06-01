#!/bin/sh

# script to run a pretrained huggingface model and eval on OffensEval 2020 English data
# Note: Set Config parameter in D4.cmd

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home2/davidyi6/ling-573/573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Preprocess train/val tweets (Greek)
python3 src/preprocess_olid.py \
    --file data/offenseval-gr-training-v1.tsv \
    --train_ids data/gr_train_ids.txt \
    --val_ids data/gr_val_ids.txt \
    --split_punctuation \
    --remove_hashtags \
    --language greek

# Preprocess test tweets (Greek)
python3 src/preprocess_olid.py \
    --file data/offenseval-gr-test-v1.tsv \
    --all_test \
    --split_punctuation \
    --remove_hashtags \
    --language greek

# Preprocess train/val tweets (English)
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv \
    --train_ids data/eng_train_ids.txt \
    --val_ids data/eng_val_ids.txt \
    --split_punctuation \
    --remove_apostraphes \
    --remove_hashtags

# Preprocess test tweets (English)
python3 src/preprocess_olid.py \
	--file data/offenseval-en-test-2020.tsv \
	--all_test \
	--split_punctuation \
	--remove_apostraphes \
	--remove_hashtags \
	--language english

# Run finetuned model predictions on test data and generate output (GREEK)
python3 src/ensemble.py \
	--val_data data/clean_val_greek.tsv \
    --test_data data/clean_all_test_greek.tsv \
	--configs configs/finetune_xlmr_large_final.json configs/finetune_xlmr_large_final.json \
	--models models/finetune_xlmr_large_final_greek models/finetune_xlmr_large_2_hybrid \
	--val_output_csv outputs/D4/adaptation/ensemble/D4_greek_preds.csv

# Run finetuned model predictions on test data and generate output (ENGLISH)
python3 src/ensemble.py \
	--val_data data/clean_val_english.tsv \
    --test_data data/clean_all_test_english.tsv \
	--configs configs/finetune_roberta.json configs/finetune_xlmr.json configs/finetune_xlmr_large_final.json \
	--models models/finetune_roberta models/finetune_xlmr_english models/finetune_xlmr_large_2_english \
	--val_output_csv outputs/D4/adaptation/ensemble/D4_english_preds.csv
