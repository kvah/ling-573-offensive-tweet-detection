#!/bin/sh

# A script to finetune a pretrained huggingface model and eval on OffensEval 2020 Greek data

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home2/davidyi6/ling-573/573_gpu

# Test whether condor is using GPU
python3 src/test_gpu.py

# # Preprocess training tweets
# python3 src/preprocess_olid.py \
#     --file data/offenseval-gr-training-v1.tsv \
#     --train_ids data/gr_train_ids.txt \
#     --val_ids data/gr_val_ids.txt \
#     --split_punctuation \
#     --remove_hashtags \
#     --language greek

# # Preprocess test tweets
# python3 src/preprocess_olid.py \
#     --file data/offenseval-gr-test-v1.tsv \
#     --all_test \
#     --split_punctuation \
#     --remove_hashtags \
#     --language greek

# # Finetune pretrained model on training data
# python3 src/finetune_pretrained.py \
#     --train_data data/clean_train_greek.tsv \
#     --val_data data/clean_val_greek.tsv \
#     --config configs/finetune_xlmr_large_final.json \
#     --train_mode greek

# Run finetuned model predictions on Greek data and generate output
python3 src/finetune_predict.py \
    --val_data data/clean_all_test_greek.tsv \
    --config configs/finetune_xlmr_large_final.json \
    --model_path models/finetune_xlmr_large_final_greek \
    --val_output_csv outputs/D4/adaptation/evaltest/D4_greek_preds.csv

# Evaluation script (Greek)
python3 src/eval.py \
    --val_output_csv outputs/D4/adaptation/evaltest/D4_greek_preds.csv \
    --output_path results/D4/adaptation/evaltest/D4_scores.out
