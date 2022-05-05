"""
    Data module (TextDataset)
"""

import vocab

import numpy as np


from vocab import Vocabulary



class OLIDDataset:
    def __init__(self, examples: list, vocabulary: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocabulary
        self.padding_index = self.vocab[vocab.PAD]

    def batch_as_tensors(self, start: int, end: int) -> dict:
        examples = [self.__getitem__(index) for index in range(start, end)]
        padding_index = self.vocab[vocab.PAD]
        return {
            "content": pad_batch(
                [example["content"] for example in examples], padding_index
            ),
            "label": np.array([example["label"] for example in examples]),
            "lengths": np.array([len(example["content"]) for example in examples]),
        }

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def pad_batch(sequences: list, padding_index: int) -> np.ndarray:
    """
    Pad a list of sequences

    Arguments:
        sequences: list of arrays, each containing integer indices, to pad and combine.
        padding_index: integer index of PAD symbol, used to fill in to make sequences longer

    Returns:
        [batch_size, max_seq_len] numpy array
    """
    # find max len 
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    # padding array
    pads = [padding_index for i in range(max_len)]
    # pad sequences
    return np.stack(
        [np.append(sequences[i], pads[lengths[i]:]) 
        for i in range(len(sequences))]
    )
