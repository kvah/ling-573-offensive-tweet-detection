import os
import sys
import argparse
import copy
from pathlib import Path 

import pandas as pd

from datasets import load_metric
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

import tqdm
from tqdm.auto import tqdm
import time

from finetune_config import Config

from sklearn.linear_model import LogisticRegression

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

if __name__ == '__main__':
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run ensemble predictions on test dataset")
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--configs", type=str, nargs='+', required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--val_output_csv", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])

    # Load validation data
    val_df = pd.read_csv(args.val_data, sep='\t')
    val_df.columns = ['content', 'labels']

    test_df = pd.read_csv(args.test_data, sep='\t')
    test_df.columns = ['content', 'labels']

    val_probs = []
    test_probs = []
    for config, model_path in zip(args.configs, args.models):
        # Load fine-tune configs
        config = Config.from_json(config)

        # Convert pandas df to huggingface datasets
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Install pretrained xlm-roberta-base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Tokenize tweets with padding/truncation
        val_tokenized = val_dataset.map(tokenize_function, batched=True)
        val_tokenized = val_tokenized.remove_columns(['content'])
        val_tokenized.set_format("torch")

        test_tokenized = test_dataset.map(tokenize_function, batched=True)
        test_tokenized = test_tokenized.remove_columns(['content'])
        test_tokenized.set_format("torch")

        val_dataloader = DataLoader(val_tokenized, batch_size=config.batch_size)
        test_dataloader = DataLoader(test_tokenized, batch_size=config.batch_size)

        metric = load_metric("f1")
        all_preds = []
        model_probs = []
        # Get val probs to train stack classifier
        model.eval()
        for batch in tqdm(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits)
                predictions = torch.argmax(logits, dim=-1)
                model_probs += probs.tolist()
                all_preds += predictions.tolist()
                metric.add_batch(predictions=predictions, references=batch["labels"])
        val_probs.append(model_probs)

        # Compute Macro f1-score
        score = metric.compute(average="macro")
        print(f'Model {model_path} Validation Macro F1 Score: {score}')

        metric = load_metric("f1")
        all_preds = []
        model_probs = []
        # Test Eval loop
        model.eval()
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits)
                predictions = torch.argmax(logits, dim=-1)
                model_probs += probs.tolist()
                all_preds += predictions.tolist()
                metric.add_batch(predictions=predictions, references=batch["labels"])
        test_probs.append(model_probs)

        # Compute Macro f1-score
        score = metric.compute(average="macro")
        print(f'Model {model_path} Test Macro F1 Score: {score}')

    # Ensemble Average Probabilities 
    val_probs = torch.tensor(val_probs)
    mean_probs = val_probs.mean(axis=0)
    mean_preds = mean_probs.argmax(dim=-1)

    metric = load_metric("f1")
    metric.add_batch(predictions=mean_preds, references=val_dataset['labels'])
    score = metric.compute(average="macro")
    print(f'Averaging Ensemble Validation Macro F1 Score: {score}')

    test_probs = torch.tensor(test_probs)
    mean_probs = test_probs.mean(axis=0)
    mean_preds = mean_probs.argmax(dim=-1)

    metric = load_metric("f1")
    metric.add_batch(predictions=mean_preds, references=test_dataset['labels'])
    score = metric.compute(average="macro")
    print(f'Averaging Ensemble Test Macro F1 Score: {score}')

    # Build Stacking Ensemble Classifier
    val_probs = torch.tensor(val_probs)
    val_features = val_probs.permute(1, 0, 2).reshape(len(val_dataset), len(args.models)*2)
    LR = LogisticRegression(fit_intercept=False)
    LR.fit(val_features, val_dataset['labels'])

    test_probs = torch.tensor(test_probs)
    test_features = test_probs.permute(1, 0, 2).reshape(len(test_dataset), len(args.models)*2)
    test_preds = LR.predict(test_features)

    metric = load_metric("f1")
    metric.add_batch(predictions=test_preds, references=test_dataset['labels'])
    score = metric.compute(average="macro")
    print(f'Stacked Ensemble Test Macro F1 Score: {score}')

