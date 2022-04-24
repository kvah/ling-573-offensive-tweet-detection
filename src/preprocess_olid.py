"""
    Module for pre-processing OLID dataset (Task A)
    
    To use as a script, execute:
        python3 preprocess_olid.py --file FILE [--lang LANG] 
        [--train_ids TRAIN_IDS --dev_ids DEV_IDS]
    
"""

import argparse
import re

import pandas as pd

from sys import argv

# define constants
OFF = "OFF"
NOT = "NOT"

# define types
Example = {"content":str, "label":int}

def preprocess(data: pd.DataFrame, lang: str="english") -> list:
    """
    Converts OLID-formatted file into a list of datapoints.

    Parameters
    ----------
    file : str
        tsv file containing data (OLID format)
    lang : str
        language of data. default is "english"

    Returns
    -------
    list of dicts: {'content': str, 'label': int}
        content is tweet
        label is 1 for offensive, 0 for not offensive

    """
    df = pd.read_csv(file, sep="\t", header=0,encoding="utf8")
    
    # cut out everything but tweet and label
    df = df[["tweet", "subtask_a"]]
    
    data = []
    
    for index, row in df.iterrows():
        if row["subtask_a"] == OFF:
            label = 1
        else:
            label = 0
        
        data.append({'content':row["tweet"], 'label':label})
        
    return(data)


def write_file(data: list, out_file: str) -> None:
    """
    Writes tweets and labels to tsv file

    """
    file = open(out_file, mode="w+", encoding="utf-8")
    # header
    file.write("content\tlabel\n")
    
    # write data
    for line in data:
        file.write(f"{line['content']}\t{line['label']}\n")
        
    file.close()
    
def split_data(data: pd.DataFrame, train_ids_file: str, 
               dev_ids_file: str) -> [pd.DataFrame, pd.DataFrame]:
    """
    Splits canonical training data into train and dev

    Parameters
    ----------
    data : pd.DataFrame
        OLID data, must include 'id' field
    train_ids_file : str
        line-separated file containing ids corresponding to training set
    dev_ids_file : str
        line-separated file containing ids corresponding to dev set

    Returns
    -------
    train_set : pd.DataFrame
    test_set : pd.DataFrame

    """
    train_ids = pd.Index([int(n) for n in 
                     open(train_ids_file).readlines()])
    dev_ids = pd.Index([int(n) for n in 
                   open(dev_ids_file).readlines()])
    
    train_set = data.loc[train_ids]
    dev_set = data.loc[dev_ids]
    
    return(train_set, dev_set)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Pre-process OLID dataset for Task A.")
    parser.add_argument(
        "--file", type=str, required=True, help="path to data (OLID format)")
    parser.add_argument(
        "--language", type=str, default="english", 
        choices=["english", "arabic", "danish","greek", "turkish"],
        help="language of data")
    parser.add_argument(
        "--train_ids", default=None, 
        help="path to file containing IDs for training set (line-separated)")
    parser.add_argument(
        "--dev_ids", default=None, 
        help="path to file containing IDs for development set (line-separated)")
    
    args = parser.parse_args(argv[1:])
    lang = args.language
    
    file = args.file
    file_ending = re.search(r"([^//]*)$", args.file).groups()[0]
    
    data = pd.read_csv(file, sep="\t", header=0, encoding="utf8")
    
    # split data if specified
    if args.train_ids != None:
        train_ids = args.train_ids
        dev_ids = args.dev_ids
        train, dev = split_data(data, train_ids, dev_ids)
        
        train_preprocessed = preprocess(train, lang)
        dev_preprocessed = preprocess(dev, lang)
        
        write_file(train_preprocessed, f"data/pp_train_{file_ending}")
        write_file(dev_preprocessed, f"data/pp_dev_{file_ending}")
        
    else:
        train = preprocess(data, lang)
        write_file(train, f"data/pp_{file_ending}")