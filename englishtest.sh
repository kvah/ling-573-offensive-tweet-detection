#!/bin/sh

# script to run a pretrained huggingface model and eval on OffensEval 2020 English data
# Note: Set Config parameter in D4.cmd

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home2/davidyi6/ling-573/573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# Preprocess tweets
python3 src/preprocess_olid.py \
	--file data/offenseval-en-test-2020.tsv \
	--all_test \
	--split_punctuation \
	--remove_apostraphes \
	--remove_hashtags \
	--language english

# Run finetuned model predictions on English test data and generate output
python3 src/finetune_predict.py \
	--val_data data/clean_all_test_english.tsv \
	--config configs/finetune_roberta.json \
	--model_path models/finetune_roberta \
	--val_output_csv outputs/D4/primary/evaltest/D4_english.csv

# Evaluation script (English)
python3 src/eval.py \
	--val_output_csv outputs/D4/primary/evaltest/D4_english.csv \
	--output_path results/D4/primary/evaltest/D4_scores.out
