"""
Get prediction outputs from trained LSTM Model
TODO: 
    - Save and Load tokenizer instead of recreating it
    - Generalize so that predictions can be generated for any dataset, not just validation set

Example Usage:
python3 src/lstm_predict.py \
    --config configs/D3.json\
    --train_data data/clean_train_olid.tsv \
    --val_data data/clean_val_olid.tsv \
    --model_config_path lstm_saved_configs/ \
    --model_path models/lstm_epoch6.pt \
    --threshold 0.5 \
    --val_output_csv outputs/D3_val_preds.csv
"""

# External dependencies
import os
import argparse
import pickle 
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from nltk.tokenize import TweetTokenizer

import vocab
from data import OLIDDataset
from lstm_model import LSTM
from lstm_config import LSTMConfig

def get_predictions_from_logits(logits: Tensor, threshold: int=0.5):
    """

    """
    preds = [1 if logit >= threshold else 0 for logit in logits]
    return preds

if __name__ == "__main__":    
    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_data",type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=False)
    parser.add_argument("--val_output_csv", type=str, required=True)
    args = parser.parse_args()

    if args.config:
        config = LSTMConfig.from_json(args.config)
    else:
        config = LSTMConfig()
    
    model_config_path = args.model_config_path
    vocab_length_path = os.path.join(model_config_path, 'vocab_length.pkl')
    padding_index_path = os.path.join(model_config_path, 'padding_index.pkl')
    embedding_matrix_path = os.path.join(model_config_path, 'embedding_matrix.npy')
    best_model_path = os.path.join('models', f'lstm_D3_best_model.pt')

    with open(vocab_length_path, 'rb') as fp:
        vocab_length = int(pickle.load(fp))
    with open(padding_index_path, 'rb') as fp:
        padding_index = int(pickle.load(fp))

    embedding_matrix = np.load(embedding_matrix_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using GPU: {device}')

    # Load model with trained params
    model_dict = args.model_path 
    model = LSTM(
        config=config, 
        vocab_length = vocab_length,
        pretrained_embs=embedding_matrix, 
        padding_idx=padding_index
    )
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_dict, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_dict))

    # TODO: Save tokenizer during training process instead of recreating it 
    train_df = pd.read_csv(args.train_data, sep="\t", header=0)
    val_df = pd.read_csv(args.val_data, sep="\t", header=0)

    # make vocab + embed matrix
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    # we want both training/val vocab here since we are not training the embeddings
    vocabulary = vocab.make_vocab(list(train_df['content']) + list(val_df['content']), tokenizer.tokenize)
    glove_embeds = embedding_matrix
    embedding_matrix, idx_to_vocab, vocab_to_idx = vocab.get_embedding_matrix(
        glove_embeds, vocabulary)
    all_vocab = vocab.Vocabulary(idx_to_vocab, vocab_to_idx)

    # convert tweets to indices
    val_df["encoded"] = val_df["content"].apply(
        lambda x: all_vocab.tokens_to_indices(tokenizer.tokenize(x)))

    # build dataset
    val_content = val_df["encoded"].to_list()
    val_labels = val_df["label"].to_list()
    val_examples = [{"content": val_content[i], "label": val_labels[i]}
                    for i in range(len(val_content))]

    olid_val_data = OLIDDataset(val_examples, all_vocab)

    with torch.no_grad():
        batch = olid_val_data.batch_as_tensors(0, len(olid_val_data))
        # Get output probabilities 
        logits = model(
            torch.LongTensor(batch["content"]), torch.LongTensor(batch["lengths"])
        )

    preds = get_predictions_from_logits(logits, 0.5)
    val_df['predicted_label'] = preds
    val_df.to_csv(args.val_output_csv)
