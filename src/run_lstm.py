"""
    Script for running LSTM classifier on OLID dataset
    
    to run:
    python3 run_lstm.py --train_data PATH --val_data PATH [--glove_embeds PATH] \
        [--dropout FLOAT] [--l2 FLOAT] [--hidden_dim INT] [--embedding_dim INT] \
        [--lr FLOAT] [--batch_size INT] [--num_epochs INT] [--seed INT] > out_file
"""

import argparse
import copy
import random
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer

import vocab
from data import OLIDDataset
from lstm_model import LSTM

if __name__ == "__main__":
    
    # TODO: figure out how to use GPU

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train_data",type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--glove_embeds", type=str,
                        default="./data/glove.twitter.27B.200d.txt")
    
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # DATA HANDLING
    train_df = pd.read_csv(args.train_data, sep="\t", header=0)
    val_df = pd.read_csv(args.val_data, sep="\t", header=0)
    
    # make vocab + embed matrix
    tokenizer = TweetTokenizer()
    vocabulary = vocab.make_vocab(train_df['content'], tokenizer.tokenize)
    glove_embeds = vocab.load_glove_vectors(args.glove_embeds)
    embedding_matrix, idx_to_vocab, vocab_to_idx = vocab.get_embedding_matrix(
        glove_embeds, vocabulary)
    train_vocab = vocab.Vocabulary(idx_to_vocab, vocab_to_idx)
    
    # convert tweets to indices
    train_df["encoded"] = train_df["content"].apply(
        lambda x:train_vocab.tokens_to_indices(tokenizer.tokenize(x)))
    val_df["encoded"] = val_df["content"].apply(
        lambda x: train_vocab.tokens_to_indices(tokenizer.tokenize(x)))
    
    # build datasets
    train_content = train_df["encoded"].to_list()
    train_labels = train_df["label"].to_list()
    train_examples = [{"content": train_content[i], "label": train_labels[i]} 
                      for i in range(len(train_content))]
    
    val_content = val_df["encoded"].to_list()
    val_labels = val_df["label"].to_list()
    val_examples = [{"content": val_content[i], "label": val_labels[i]}
                    for i in range(len(val_content))]
    
    olid_train_data = OLIDDataset(train_examples, train_vocab)
    olid_val_data = OLIDDataset(val_examples, train_vocab)
    
    
    # dev data as np arrays
    val_data = olid_val_data.batch_as_tensors(0, olid_val_data.__len__())

    # BUILD MODEL
    padding_index = olid_train_data.padding_index
    
    model = LSTM(vocab_length = train_vocab.__len__(), pretrained_embs=embedding_matrix,
                 hidden=args.hidden_dim, drop_out= args.dropout,
                 padding_idx=padding_index)

    # get training things set up
    data_size = olid_train_data.__len__()
    batch_size = args.batch_size
    starts = list(range(0, data_size, batch_size))
    # TODO: look into optimizer args
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.l2, lr=args.lr)
    best_loss = float("inf")
    best_model = None
    loss_fn = torch.nn.BCELoss()

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # shuffle batches
        random.shuffle(starts)
        for start in tqdm(starts):
            batch = olid_train_data.batch_as_tensors(
                start, min(start + batch_size, data_size)
            )
            # get probabilities and loss
            # [batch_size, num_labels]
            model.train()
            logits = model(
                torch.LongTensor(batch["content"]), torch.LongTensor(batch["lengths"])
            )
            
            loss = loss_fn(
                torch.squeeze(logits), torch.FloatTensor(batch["label"])
            )
            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train loss: {running_loss / len(starts)}")

        # get dev loss every epoch
        model.eval()
        logits = model(
            torch.LongTensor(val_data["content"]), torch.LongTensor(val_data["lengths"])
        )
        epoch_loss = loss_fn(
            torch.squeeze(logits), torch.FloatTensor(val_data["label"])
        ).item()
        print(f"Epoch {epoch} dev loss: {epoch_loss}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)

    # TODO: report dev f1