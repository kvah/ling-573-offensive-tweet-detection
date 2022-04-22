#!/bin/sh

python3 preprocess_olid.py --file olid-training-v1.0.tsv
python3 -m gensim.scripts.glove2word2vec --input glove.twitter.27B.200d.txt --output glove.twitter.27B.200d.w2vformat.txt
python3 featurize_tweets.py --preprocessed_data pp_olid-training-v1.0.tsv --train_indices train_ids.txt --val_indices val_ids.txt --embedding_path glove.twitter.27B.200d.w2vformat.txt --embedding_size 200
python3 classify_tweets.py --preprocessed_data pp_olid-training-v1.0.tsv --train_indices train_ids.txt --val_indices val_ids.txt --train_vectors train_vectors.npy --val_vectors val_vectors.npy
