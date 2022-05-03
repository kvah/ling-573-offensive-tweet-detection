import sys
import argparse
from typing import List, Dict 

import numpy as np 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

from utils import df_from_indices

def create_embedding_matrix(
    embedding_path:str, 
    embedding_size:int,
    word_to_index:Dict, 
    max_features:int
) -> np.ndarray:
    """Load pretrained embedding"""
    word_embeddings = KeyedVectors.load_word2vec_format(embedding_path, binary=False)

    embedding_matrix = np.zeros((max_features, embedding_size))
    for word, idx in word_to_index.items():
        if idx < max_features and word in word_embeddings:
            embedding_matrix[idx] = word_embeddings[word]
    return embedding_matrix 

def embeddings_from_sequences(padded_sequence: np.ndarray, embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Input: 
        padded_sequence: numpy array of shape (num_samples, seq_len)
    Output: 
        Feature embedding array of shape (num_samples, embedding_size)
    """
    
    feature_embeddings = []
    for seq in padded_sequence:
        embedding = np.concatenate([embedding_matrix[seq][i] for i in range(len(seq))])
        feature_embeddings.append(embedding)
    feature_embeddings = np.array(feature_embeddings)
    return feature_embeddings


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create embedding features from a .tsv file containing tweets")
    parser.add_argument("--preprocessed_data", type=str, required=True)
    parser.add_argument("--train_indices", type=str, required=True)
    parser.add_argument("--val_indices", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--embedding_size", type=int, required=True)
    parser.add_argument("--train_vectors", type=str, default='data/train_vectors.npy')
    parser.add_argument("--val_vectors", type=str, default='data/val_vectors.npy')

    args = parser.parse_args(sys.argv[1:])

    # Load training and validation data
    tweets = pd.read_csv(args.preprocessed_data, sep='\t')
    train_tweets = df_from_indices(args.train_indices, tweets)
    val_tweets = df_from_indices(args.val_indices, tweets)

    # Fit tokenizer on all unique tokens
    all_tweets = list(train_tweets.content) + list(val_tweets.content)
    tk = Tokenizer(lower=True, filters='')
    tk.fit_on_texts(all_tweets)

    # Pad training/validation sequences
    max_len = 50
    train_tokens = tk.texts_to_sequences(train_tweets.content)
    val_tokens = tk.texts_to_sequences(val_tweets.content)
    train_padded = pad_sequences(train_tokens, maxlen = max_len)
    val_padded = pad_sequences(val_tokens, maxlen = max_len)

    # Load embeddings and featurize sequences
    word_to_index = tk.word_index
    word_embeddings = create_embedding_matrix(
        embedding_path=args.embedding_path, 
        embedding_size=args.embedding_size,
        max_features=min(len(word_to_index)+1, 40000),
        word_to_index=word_to_index
    )
    train_embeddings = embeddings_from_sequences(train_padded, word_embeddings)
    val_embeddings = embeddings_from_sequences(val_padded, word_embeddings)

    # Save train and validation feature vectors
    np.save(args.train_vectors, train_embeddings)
    np.save(args.val_vectors, val_embeddings)
