"""
    Script for running LSTM classifier on OLID dataset
    
    to run:
    python3 run_lstm.py --config PATH --train_data PATH --val_data PATH --model_config_path PATH  > out_file
"""

import os
import pickle
import argparse
import copy
import random
import torch
import re

import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk.tokenize import TweetTokenizer
import time

import vocab
from data import OLIDDataset
from lstm_model import LSTM
from lstm_config import LSTMConfig

if __name__ == "__main__":    
    start_time = time.time()

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train_data",type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--train_tokens_only", action="store_true")

    args = parser.parse_args()
    
    # Enable cuda if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using Device: {device}')
    
    # make config
    if args.config:
        config = LSTMConfig.from_json(args.config)
        config_name = re.search(r"([^//]*).json", args.config).groups(1)[0]
    else:
        # use default values if no config given
        config = LSTMConfig()
        config_name = str(time.time())

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # DATA HANDLING
    train_df = pd.read_csv(args.train_data, sep="\t", header=0)
    val_df = pd.read_csv(args.val_data, sep="\t", header=0)
    
    # make vocab + embed matrix
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    if args.train_tokens_only:
        vocabulary = vocab.make_vocab(list(train_df['content']), tokenizer.tokenize)
    else:
        vocabulary = vocab.make_vocab(list(train_df['content']) + list(val_df['content']), tokenizer.tokenize)
    glove_embeds = vocab.load_glove_vectors(config.glove_embeds)
    embedding_matrix, idx_to_vocab, vocab_to_idx = vocab.get_embedding_matrix(glove_embeds, vocabulary)
    all_vocab = vocab.Vocabulary(idx_to_vocab, vocab_to_idx)
    
    # convert tweets to indices
    train_df["encoded"] = train_df["content"].apply(
        lambda x: all_vocab.tokens_to_indices(tokenizer.tokenize(x)))
    val_df["encoded"] = val_df["content"].apply(
        lambda x: all_vocab.tokens_to_indices(tokenizer.tokenize(x)))
    
    # build datasets
    train_content = train_df["encoded"].to_list()
    train_labels = train_df["label"].to_list()
    train_examples = [{"content": train_content[i], "label": train_labels[i]} 
                      for i in range(len(train_content))]
    
    val_content = val_df["encoded"].to_list()
    val_labels = val_df["label"].to_list()
    val_examples = [{"content": val_content[i], "label": val_labels[i]}
                    for i in range(len(val_content))]
    
    olid_train_data = OLIDDataset(train_examples, all_vocab)
    olid_val_data = OLIDDataset(val_examples, all_vocab)
    
    
    # dev data as np arrays
    val_data = olid_val_data.batch_as_tensors(0, olid_val_data.__len__())

    # BUILD MODEL
    padding_index = olid_train_data.padding_index
    
    model = LSTM(
        config=config, 
        vocab_length = all_vocab.__len__(), 
        pretrained_embs=embedding_matrix, 
        padding_idx=padding_index
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
    # Create directory for saved models
    os.makedirs('models', exist_ok=True)
    
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
            model_path = f'models/lstm_epoch{epoch}.pt'
            print(f"Saving model to : {model_path}")
            torch.save(best_model.state_dict(), model_path)
    
    # Save model configs so they can be loaded for prediction
    model_config_path = args.model_config_path
    os.makedirs(model_config_path, exist_ok=True)

    vocab_length_path = os.path.join(model_config_path, 'vocab_length.pkl')
    padding_index_path = os.path.join(model_config_path, 'padding_index.pkl')
    embedding_matrix_path = os.path.join(model_config_path, 'embedding_matrix.npy')
    best_model_path = os.path.join('models', f'lstm_{config_name}_best_model.pt')

    with open(vocab_length_path, 'wb') as fp:
        pickle.dump(f'{best_model.vocab_length}', fp)
    with open(padding_index_path, 'wb') as fp:
        pickle.dump(f'{best_model.padding_idx}', fp)

    np.save(embedding_matrix_path, best_model.embeddings.weight.cpu().data.numpy())
    torch.save(best_model.state_dict(), best_model_path)
    print(f'Best model saved to: {best_model_path}')
    
    csv.close()        

    print(f'Training Time Elapsed: {time.time() - start_time} seconds')
