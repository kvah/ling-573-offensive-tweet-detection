"""
    Script for running LSTM classifier on OLID dataset
    
    to run:
    python3 run_lstm.py --config PATH --train_data PATH --val_data PATH  > out_file
"""

import argparse
import copy
import random
import torch
import re

import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
from time import time

import vocab
from data import OLIDDataset
from lstm_model import LSTM
from lstm_config import LSTMConfig

if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_data",type=str)
    parser.add_argument("--val_data", type=str)
    
    args = parser.parse_args()
    
    # make config
    if args.config:
        config = LSTMConfig.from_json(args.config)
        config_name = re.search(r"([^//]*).json", config).groups(1)
    else:
        # use default values if no config given
        config = LSTMConfig()
        config_name = str(time())

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # DATA HANDLING
    train_df = pd.read_csv(args.train_data, sep="\t", header=0)
    val_df = pd.read_csv(args.val_data, sep="\t", header=0)
    
    # make vocab + embed matrix
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    vocabulary = vocab.make_vocab(train_df['content'], tokenizer.tokenize)
    glove_embeds = vocab.load_glove_vectors(config.glove_embeds)
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

    # Enable cuda if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using GPU: {device}')
    
    model = LSTM(
        config=config, vocab_length = train_vocab.__len__(), 
        pretrained_embs=embedding_matrix, padding_idx=padding_index
    )
    if device.type == 'cuda':
        model = model.cuda()

    # get training things set up
    data_size = olid_train_data.__len__()
    batch_size = config.batch_size
    starts = list(range(0, data_size, batch_size))
    # TODO: look into optimizer args
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=config.l2, lr=config.lr)
    best_loss = float("inf")
    best_model = None
    loss_fn = torch.nn.BCELoss()
    
    # record train, dev loss in a csv
    csv = open(f"{config_name}_lstm.csv", mode="w+")
    csv.write("epoch,train_loss,dev_loss\n")

    for epoch in tqdm(range(config.num_epochs)):
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
                torch.LongTensor(batch["content"]).to(device), torch.LongTensor(batch["lengths"]).to(device)
            )
            
            loss = loss_fn(
                torch.squeeze(logits), torch.FloatTensor(batch["label"]).to(device)
            )
            running_loss += loss.item()

            # get gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = running_loss / len(starts)
        print(f"Epoch {epoch} train loss: {train_loss}")

        # get dev loss every epoch
        model.eval()
        logits = model(
            torch.LongTensor(val_data["content"]).to(device), torch.LongTensor(val_data["lengths"]).to(device)
        )
        epoch_loss = loss_fn(
            torch.squeeze(logits), torch.FloatTensor(val_data["label"]).to(device)
        ).item()
        print(f"Epoch {epoch} dev loss: {epoch_loss}")
        csv.write(f"{epoch},{train_loss},{epoch_loss}\n")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New best loss; saving current model")
            best_model = copy.deepcopy(model)
            # Save params of best model
            torch.save(best_model.state_dict(), f'lstm_{config_name}_epoch{epoch}.pt')
    
    csv.close()        
    # TODO: report dev f1