"""
    LSTM classifier for offensive language detection
"""

import torch

import numpy as np
import torch.nn as nn

from torch import Tensor
from lstm_config import LSTMConfig

class LSTM(nn.Module):

    def __init__(self, config: LSTMConfig, vocab_length: int, pretrained_embs: np.ndarray,
                 padding_idx: int = 0):
        super(LSTM, self).__init__()
        
        self.vocab_length = vocab_length
        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(
            vocab_length, config.embedding_dim, padding_idx=self.padding_idx
            )
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embs))
        # freeze embeddings
        if config.freeze_embeds:
            self.embeddings.weight.requires_grad = False
        
        
        self.hidden = config.hidden_dim
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                            hidden_size=self.hidden,
                            num_layers=config.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=config.dropout)

        self.linear = nn.Linear(self.hidden, 1)

    def forward(self, sequences: Tensor, lengths: Tensor):

        # [batch_size, seq_len, embed_dim]
        embeds = self.embeddings(sequences)
        embeds = self.drop(embeds)
        
        # output: [batch_size, seq_len, hidden_dim]
        output, (ht, ct) = self.lstm(embeds)
        # [batch_size, seq_len, 1]
        out = torch.sigmoid(self.linear(ht[-1]))

        return out