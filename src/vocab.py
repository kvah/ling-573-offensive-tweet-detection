# -*- coding: utf-8 -*-
"""
    Vocab module for loading pretrained embeddings
"""

import numpy as np

from typing import Iterable
from collections import Counter
from gensim.models import KeyedVectors

# define constants
PAD = "<PAD>"
UNK = "<UNK>"

class Vocabulary:
    """
        Stores a bidirectional mapping of tokens to integer indices.
    """

    def __init__(
        self,
        idx_to_tokens: list,
        tokens_to_idx: dict,
    ) -> None:

        self.index_to_token = idx_to_tokens
        self.token_to_index = tokens_to_idx

    def __len__(self) -> int:
        """Get length of the vocab. """
        return len(self.index_to_token)

    def __getitem__(self, token: str) -> int:
        """Get the index of a token.
        Returns the index of <unk> if there is an unk token and the token is not in vocab.
        Raises a ValueError if there is no unk token and the token is not in vocab. """
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return self.token_to_index[UNK]


    def tokens_to_indices(self, tokens: Iterable[str]) -> list:
        """Get all indices for a list of tokens. """
        return [self.__getitem__(t) for t in tokens]

    def indices_to_tokens(self, indices: Iterable[int]) -> list:
        """Get all tokens for a list of integer indices. """
        return [self.index_to_token[i] for i in indices]



def make_vocab(text: list, tokenizer) -> Counter:
    """
    Creates Counter of words in data

    Parameters
    ----------
    text : list[str]
    tokenizer : function that returns list

    """
    counts = Counter()
    
    for utterance in text:
        counts.update(tokenizer(utterance))
        
    return(counts)
    


def load_glove_vectors(glove_file="./data/glove.twitter.27B.200d.w2vformat.txt")-> dict:
    """
        Load the glove word vectors
    """
    word_embeddings = KeyedVectors.load_word2vec_format(glove_file, binary=False)
    return word_embeddings


def get_embedding_matrix(word_vecs: dict, vocab: list, 
                         emb_size: int = 200) -> (np.ndarray, list, dict):
    """ 
        Creates embedding matrix from word vectors
        
        Parameters
        ----------
        word_vecs: dict -- str: np.ndarray
            pretrained word embeddings
        vocab: list
            words in data
        embed_size: int
            length of embeddings
            
        Returns
        ---------
        np.ndarray : embedding matrix
        list : idx_to_vocab
        dict : vocab_to_idx
            
    """
    
    vocab_size = len(vocab) + 2
    vocab_to_idx = {}
    embed_matrix = np.zeros((vocab_size, emb_size), dtype="float32")
    
    # pad
    embed_matrix[0] = np.zeros(emb_size, dtype='float32')
    # unk
    # embed_matrix[1] = np.random.uniform(-0.25, 0.25, emb_size)
    embed_matrix[1] = np.zeros(emb_size, dtype='float32')
    
    idx_to_vocab = [PAD,UNK]
    vocab_to_idx[PAD] = 0
    vocab_to_idx[UNK] = 1
    
    i = 2
    for word in vocab:
        if word in word_vecs:
            embed_matrix[i] = word_vecs[word]
        else:
            embed_matrix[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        idx_to_vocab.append(word)
        i += 1   
    return embed_matrix, idx_to_vocab, vocab_to_idx

