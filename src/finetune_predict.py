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

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True)

if __name__ == '__main__':
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run predictions & eval on validation/test set")
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_output_csv", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])

    # Load validation data
    val_df = pd.read_csv(args.val_data, sep='\t')
    val_df.columns = ['content', 'labels']

    # Load fine-tune configs
    config = Config.from_json(args.config)
    config_name = Path(args.config).stem

    # Convert pandas df to huggingface datasets
    val_dataset = Dataset.from_pandas(val_df)

    # Install pretrained xlm-roberta-base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Tokenize tweets with padding/truncation
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_tokenized.remove_columns(['content'])
    val_tokenized.set_format("torch")

    val_dataloader = DataLoader(val_tokenized, batch_size=config.batch_size)
    metric = load_metric("f1")

    all_preds = []
    # Eval loop
    model.eval()
    for batch in tqdm(val_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_preds += predictions.tolist()
            metric.add_batch(predictions=predictions, references=batch["labels"])

    # Compute Macro f1-score
    score = metric.compute(average="macro")
    print(f'Validation Macro F1 Score: {score}')

    val_df['predicted_label'] = all_preds
    val_df['label'] = val_df['labels']
    val_df = val_df[['label', 'predicted_label', 'content']]
    output_dir = os.path.dirname(args.val_output_csv)

    os.makedirs(output_dir, exist_ok=True)
    val_df.to_csv(args.val_output_csv)

    print(f'Time Elapsed (Inference): {time.time() - start_time}')
