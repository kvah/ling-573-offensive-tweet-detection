"""
Module for classifying BERT embeddings (Task A)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import df_from_indices


def get_labels(indices_path: str, tweets: pd.DataFrame) -> np.ndarray:
    """
    Purpose: Get true labels from preprocessed data
    Input: [indices_path]: file path to indices file (str)
           [tweets]: pandas dataframe that stores the tweets (pd.DataFrame)
    Output: true labels (np.ndarray)
    """
    tweets = df_from_indices(indices_path, tweets)
    return tweets['label'].values


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Classify featurized tweets with logistic regression algorithm")
    parser.add_argument("--preprocessed_data", type=str, required=True)
    parser.add_argument("--train_indices", type=str, required=True)
    parser.add_argument("--val_indices", type=str, required=True)
    parser.add_argument("--train_vectors", type=str, required=True)
    parser.add_argument("--val_vectors", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])

    # load tweets
    tweets = pd.read_csv(args.preprocessed_data, sep="\t", header=0, encoding="utf8")

    # get training and validation true labels
    train_true_labels = get_labels(args.train_indices, tweets)
    val_true_labels = get_labels(args.val_indices, tweets)

    # load training and validation vectors
    train_vec = np.load(args.train_vectors)
    val_vec = np.load(args.val_vectors)

    # build classifier with logistic regression algorithm
    classifier = LogisticRegression(max_iter=1000, class_weight="balenced").fit(train_vec, train_true_labels)
    
    # classify validation dataset and print out results/report
    val_pred_labels = classifier.predict(val_vec)

    print("Confusion Matrix")
    print(confusion_matrix(val_true_labels, val_pred_labels))
    print()
    print("Accuracy")
    print(accuracy_score(val_true_labels, val_pred_labels))
    print()
    print("Report")
    print(classification_report(val_true_labels, val_pred_labels))

