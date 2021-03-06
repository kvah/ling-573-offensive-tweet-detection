"""
Module for classifying BERT embeddings (Task A)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--train_vectors", type=str, required=True)
    parser.add_argument("--val_vectors", type=str, required=True)
    parser.add_argument("--val_output_csv", type=str, required=False)

    args = parser.parse_args(sys.argv[1:])

    # load tweets
    train_tweets = pd.read_csv(args.train_data, sep='\t')
    val_tweets = pd.read_csv(args.val_data, sep='\t')

    # get training and validation true labels
    train_true_labels = train_tweets['label']
    val_true_labels = val_tweets['label']

    # load training and validation vectors
    train_vec = np.load(args.train_vectors)
    val_vec = np.load(args.val_vectors)

    # build classifier with logistic regression algorithm
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced").fit(train_vec, train_true_labels)
    
    # classify validation dataset and print out macro f1-score
    val_pred_labels = classifier.predict(val_vec)
    if args.val_output_csv:
        # Save validatition data and predictions
        val_tweets['predicted_label'] = val_pred_labels
        val_tweets.to_csv(args.val_output_csv)

    print("macro f1-score")
    print(f1_score(val_true_labels, val_pred_labels, average="macro"))
    
    # build and print confusion matrix
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for a in range(len(val_pred_labels)):
        if val_pred_labels[a] == 1 and val_true_labels[a] == 1:
            tp += 1
        if val_pred_labels[a] == 1 and val_true_labels[a] == 0:
            fp += 1
        if val_pred_labels[a] == 0 and val_true_labels[a] == 0:
            tn += 1
        if val_pred_labels[a] == 0 and val_true_labels[a] == 1:
            fn += 1
    print("confusion matrix")
    confusionmatrix = ""
    confusionmatrix += "           predicted label\n"
    confusionmatrix += "true label negative    positive\n"
    confusionmatrix += "negative   "+str(tn)+"         "+str(fp)+"\n"
    confusionmatrix += "positive   "+str(fn)+"         "+str(tp)+"\n"
    print(confusionmatrix)

