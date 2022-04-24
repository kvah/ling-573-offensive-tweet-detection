# Evaluation script

import argparse 
import sys 
import numpy as np
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Write Evaluation Score (Macro F1) to output file")
    parser.add_argument("--preds_path", type=str, required=True)
    parser.add_argument("--true_labels_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    val_pred_labels = np.load(args.preds_path)
    val_true_labels = np.load(args.true_labels_path)
    f1_score = f1_score(val_true_labels, val_pred_labels, average="macro")
    print(f'Macro F1 Score: {f1_score}')

    with open(args.output_path, 'w') as fp:
        fp.write(f'Macro F1 Score: {f1_score}')
