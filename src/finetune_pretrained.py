import os
import sys
import argparse
import copy
from pathlib import Path 

import pandas as pd
import numpy as np

from datasets import load_metric
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

from tqdm.auto import tqdm
import time

from finetune_config import Config

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

if __name__ == '__main__':
    start_time = time.time()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Finetune Pretrained Model on Preprocessed Tweets")
    parser.add_argument("--train_data", type=str, nargs='+', required=True)
    parser.add_argument("--val_data", type=str, nargs='+', required=True)
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    
    # Load fine-tune configs
    config = Config.from_json(args.config)
    config_name = Path(args.config).stem
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load training data
    train_sets = []
    for file in args.train_data:
        train_set = pd.read_csv(file, sep='\t')
        train_set.columns = ['content', 'labels']
        train_sets.append(train_set)
        
    # concatenate and shuffle all training data
    train_df = pd.concat(train_sets, ignore_index=True).sample(frac=1)
    # convert to huggingface dataset
    train_dataset = Dataset.from_pandas(train_df)
    
    # load validation data and convert to huggingface dataset
    val_datasets = {}
    for file in args.val_data:
        val_set = pd.read_csv(file, sep='\t')
        val_set.columns = ['content', 'labels']
        val_datasets[file] = Dataset.from_pandas((val_set))

    # Install pretrained xlm-roberta-base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=2)

    # Tokenize tweets with padding/truncation
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    if config.num_samples:
        train_tokenized = train_tokenized.select(range(config.num_samples))
    train_tokenized = train_tokenized.remove_columns(['content', '__index_level_0__'])
    
    train_tokenized.set_format("torch")
    train_dataloader = DataLoader(train_tokenized, shuffle=True, batch_size=config.batch_size)
    
    val_tokenized = {}
    val_dataloaders = {}
    for name in val_datasets.keys():
        ds = val_datasets[name]
        val_set = ds.map(tokenize_function, batched=True)
        val_set = val_set.remove_columns(['content'])
        val_set.set_format("torch")
        
        val_tokenized[name] = val_set
        val_dataloaders[name] = DataLoader(val_set, batch_size=config.batch_size)
    

    # Training Parameters
    optimizer = AdamW(model.parameters(), lr=config.lr)
    num_epochs = config.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=config.num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    metric = load_metric("f1")

    # Training Loop
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    progress_bar = tqdm(range(num_training_steps))

    best_model, best_score = None, -float('inf')
    best_epoch = None
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            running_loss += loss.item()
            loss.backward()

            # Update gradients
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        train_loss = running_loss / len(batch)
        print(f"Epoch {epoch} train loss: {train_loss}")

        # Eval loop
        model.eval()
        for name in val_dataloaders.keys():
            val_dataloader = val_dataloaders[name]
            per_file_metric = load_metric("f1")
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(
                        predictions=predictions, 
                        references=batch["labels"]
                        )
                    per_file_metric.add_batch(
                        predictions=predictions,
                        references=batch['labels']
                        )
                    
            # compute file-specific macro f1
            file_specific_score = per_file_metric.compute(average='macro')
            print(f"Epoch {epoch} Macro F1 Score for {name}: {file_specific_score['f1']}")
                    

        # Compute Overall Macro f1-score
        score = metric.compute(average="macro")
        print(f'Epoch {epoch} overall Macro F1 Score: {score["f1"]}')

        if score['f1'] > best_score:
            print(f'Saving best model at epoch: {epoch}')
            best_score = score['f1']
            best_model = copy.deepcopy(model)
            best_epoch = epoch

    # Save fine-tuned model
    os.makedirs('models', exist_ok=True)
    best_model.save_pretrained(f'models/{config_name}')

    print(f'Time Elapsed (Finetuning): {time.time() - start_time}')
