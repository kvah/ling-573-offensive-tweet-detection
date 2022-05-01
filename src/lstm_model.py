"""
    LSTM classifier for offensive language detection
"""

import torch

import numpy as np
import torch.nn as nn

from torch import Tensor

class LSTM(nn.Module):

    def __init__(self, vocab_length: int, pretrained_embs: np.ndarray,
                 hidden: int = 300, num_layers: int = 1, drop_out: float = 0.0,
                 embed_size: int = 200, padding_idx: int = 0,
                 freeze_embeddings: bool = False):
        super(LSTM, self).__init__()

        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(vocab_length, embed_size, padding_idx=self.padding_idx)
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embs))
        # freeze embeddings
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False
        
        
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=drop_out)

        self.linear = nn.Linear(hidden, 1)

    def forward(self, sequences: Tensor, lengths: Tensor):

        # [batch_size, seq_len, embed_dim]
        embeds = self.embeddings(sequences)
        embeds = self.drop(embeds)
        
        # output: [batch_size, seq_len, hidden_dim]
        output, (ht, ct) = self.lstm(embeds)
        # [batch_size, seq_len, 1]
        out = torch.sigmoid(self.linear(ht[-1]))

        return out