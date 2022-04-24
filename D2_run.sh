#!/bin/sh

# Preprocess tweets
python3 src/preprocess_olid.py \
    --file data/olid-training-v1.0.tsv

# Convert glove embeddings to word2vec format
python3 -m gensim.scripts.glove2word2vec \
    --input data/glove.twitter.27B.200d.txt \
    --output data/glove.twitter.27B.200d.w2vformat.txt

# Create feature embeddings for train/val tweets
python3 src/featurize_tweets.py \
    --preprocessed_data data/pp_olid-training-v1.0.tsv \
    --train_indices data/train_ids.txt \
    --val_indices data/val_ids.txt \
    --embedding_path data/glove.twitter.27B.200d.w2vformat.txt \
    --embedding_size 200 \
    --train_vectors data/train_vectors.npy \
    --val_vectors data/val_vectors.npy

# Binary classification of train/val tweets as offensive/non-offensive
python3 src/classify_tweets.py \
    --preprocessed_data data/pp_olid-training-v1.0.tsv \
    --train_indices data/train_ids.txt \
    --val_indices data/val_ids.txt \
    --train_vectors data/train_vectors.npy \
    --val_vectors data/val_vectors.npy \
    --preds_path outputs/D2_preds.npy \
    --true_labels_path data/D2_true_labels.npy

# Evaluation script
python3 src/eval.py \
    --preds_path outputs/D2_preds.npy \
    --true_labels_path data/D2_true_labels.npy \
    --output_path results/D2_scores.out