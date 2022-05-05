"""
Module for resampling: Resample before preprocessing
Resample Strategy: Oversample/Undersample

To use as a script, execute:
    python3 resample.py --data olid-training-v1.0.tsv --train_ids train_ids.txt 
    --resample_strategy STRATEGY --alpha FLOAT

--resample_strategy: undersample / oversample

"""
import argparse
import sys
import numpy as np
import pandas as pd
from utils import df_from_indices

def undersample(alpha: float, maj_class_ids: pd.DataFrame, min_class_len: int) -> np.ndarray:
    """
    Undersample majority class: Nr_maj = alpha * N_min
    
    Parameters
    ----------
    alpha: float
        the ratio used for undersampling
    maj_class_ids: pd.DataFrame
        the ids of the samples in the majority class
    min_class_len: int
        the number of samples in the minority class

    Returns
    -------
    maj_class_ids: np.ndarray: 
        numpy array containing the ids of majority class after undersampling

    """
    Nr_maj = int(alpha * min_class_len)
    return maj_class_ids.sample(n = Nr_maj, random_state=42)[0].values

def oversample(alpha: float, min_class_ids: pd.DataFrame, maj_class_len: int) -> np.ndarray:
    """
    Oversample minority class: Nr_min = alpha * N_maj
    
    Parameters
    ----------
    alpha: float
        the ratio used for undersampling
    min_class_ids: pd.DataFrame
        the ids of the samples in the minority class
    maj_class_len: int
        the number of samples in the majority class
    
    Returns
    -------
    min_class_ids: np.ndarray: 
        numpy array containing the ids of minority class after oversampling

    """
    min_class_len = len(min_class_ids)
    Nr_min = int(alpha * maj_class_len)
    
    if Nr_min > min_class_len:
        add_sample_size = Nr_min - min_class_len # numbers of duplicate/added samples
        add_sample_ids = min_class_ids.sample(n = add_sample_size, random_state=42, replace = True)[0].values
        min_class_ids = np.concatenate([min_class_ids[0].values, add_sample_ids])
    else:
        sample_size = Nr_min
        min_class_ids = min_class_ids.sample(n = sample_size, random_state=42, replace = True)[0].values
    
    return min_class_ids


def get_train_tweets(train_ids_file: str, olid_data_file: str) -> pd.DataFrame:
    """
    Get training instances from the original OLID dataset
    
    Parameters
    ----------
    train_ids_file: str
        file containing training set ids
    olid_data_file: str
        OLID dataset
    
    Returns
    -------
    train_tweets: pd.DataFrame: 
        data frame containing training dataset

    """
    tweets = pd.read_csv(olid_data_file, sep="\t", header=0, encoding="utf8")
    train_tweets = df_from_indices(train_ids_file, tweets)
    return train_tweets

def get_class_ids(tweets: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """
    Get the ids of each class
    
    Parameters
    ----------
    tweets: pd.DataFrame
        data frame containg tweets
    
    Returns
    -------
    min_class.index: pd.DataFrame: 
        data frame containing the ids of minority class
    maj_class.index: pd.DataFrame: 
        data frame containing the ids of majority class

    """
    off_tweets = tweets.loc[tweets['subtask_a'] == "OFF"]
    not_tweets = tweets.loc[tweets['subtask_a'] == "NOT"]
    
    if len(off_tweets) < len(not_tweets):
        min_class = off_tweets
        maj_class = not_tweets
    else:
        min_class = not_tweets
        maj_class = off_tweets
        
    return pd.DataFrame(min_class.index), pd.DataFrame(maj_class.index)

def output_ids(ids: list, output_file: str) -> None:
    """
    Writes resampled training data ids to file
    """

    with open(output_file, "w", encoding = "utf8") as f:
        for id in ids:
            f.write("%d\n"%(id))
    f.close()

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Resample training dataset")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--train_ids", type=str, required=True)
    parser.add_argument("--resample_strategy", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)

    args = parser.parse_args(sys.argv[1:])

    # get training dataset
    train_tweets = get_train_tweets(args.train_ids, args.data)

    # get ids of minority and majority classes
    min_class_ids, maj_class_ids = get_class_ids(train_tweets)

    # resample
    if args.resample_strategy == "oversample":
        min_class_ids = oversample(args.alpha, min_class_ids, len(maj_class_ids))
        maj_class_ids = maj_class_ids[0].values
    elif args.resample_strategy == "undersample":
        maj_class_ids = undersample(args.alpha, maj_class_ids, len(min_class_ids))
        min_class_ids = min_class_ids[0].values

    # get all ids for training data
    resampled_ids = list(min_class_ids) + list(maj_class_ids)

    # output ids
    output_fp = "data/" + args.resample_strategy + "_train_ids.txt"
    output_ids(resampled_ids, output_fp)

