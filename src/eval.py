# Evaluation script

import argparse 
import sys 
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import time

if __name__ == "__main__":
    start_time = time.time() 

    # parse arguments
    parser = argparse.ArgumentParser(description="Write Evaluation Score (Macro F1) to output file")
    parser.add_argument("--val_output_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    val_df = pd.read_csv(args.val_output_csv)
    val_pred_labels = val_df['predicted_label']
    val_true_labels = val_df['label']
    f1_score = f1_score(val_true_labels, val_pred_labels, average="macro")
    print(f'Macro F1 Score: {f1_score}')

    with open(args.output_path, 'w') as fp:
        fp.write(f'Macro F1 Score: {f1_score}')

    print(f'Evaluation Time Elapsed: {time.time() - start_time} seconds')
