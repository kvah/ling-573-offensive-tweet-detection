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
import emoji

# define constants
OFF = "OFF"
NOT = "NOT"

# define types
Example = {"content":str, "label":int}

def split_punctuation(content: pd.Series) -> pd.Series:
    """
    Add whitespace seperation between and punctuation. 

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way , I don't agree with your argument. 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    cleaned_content = content.replace({r'(\w+)([!?.,])': r'\1 \2'}, regex=True)
    return content 

def remove_apostraphes(content: pd.Series) -> pd.Series:
    """
    Remove apostraphe from all contractions

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I dont agree with your argument. #livid 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    content = content.replace({r"(\w+)[’'](\w+)": r'\1\2'}, regex=True)
    return content

def remove_hashtags(content: pd.Series) -> pd.Series:
    """
    Remove hashtags from their attached phrase

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I don't agree with your argument. livid 👊👊"
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    content = content.replace({r"#(\w+)": r'\1'}, regex=True)
    return content

def split_emojis(content: pd.Series) -> pd.Series:
    """
    Add whitespace seperation between consecutive emojis

    Example: 
        tweet: "By the way, I don't agree with your argument. #livid 👊👊"
        new tweet: "By the way, I don't agree with your argument. #livid  👊  👊 "
    
    Parameters
    ----------
    content : pd.Series
        pandas Series containing tweets

    Returns
    -------
    content: pd.Series: 
        pandas Series containing cleaned tweets

    """
    emoji_with_space = {emo: f' {emo} ' for emo in emoji.UNICODE_EMOJI}
    for emo, emo_spaced in emoji_with_space.items():
        try:
            content = content.str.replace(emo, emo_spaced, regex=True)
        except Exception:
            print(f'Failed to parse emoji: {emo}')
    return content

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
    
    # cut out everything but tweet and label
    tweets = data[["tweet", "subtask_a"]]

    # preprocessing tasks
    if args.split_punctuation:
        tweets['tweet'] = split_punctuation(tweets['tweet'])
    if args.remove_apostraphes:
        tweets['tweet'] = remove_apostraphes(tweets['tweet'])
    if args.remove_hashtags:
        tweets['tweet'] = remove_hashtags(tweets['tweet'])
    if args.split_emojis:
        tweets['tweet'] = split_emojis(tweets['tweet'])
    
    data_list = []
    
    for index, row in tweets.iterrows():
        if row["subtask_a"] == OFF:
            label = 1
        else:
            label = 0
        
        data_list.append({'content':row["tweet"], 'label':label})
        
    return(data_list)


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

    parser.add_argument(
        "--split_punctuation", type=bool, default=True,
        help="whether to split punctuation from the end of tokens"
    )
    parser.add_argument(
        "--remove_apostraphes", type=bool, default=True,
        help="whether to remove apostraphe from contractions"
    )
    parser.add_argument(
        "--remove_hashtags", type=bool, default=False,
        help="whether to remove the # symbol from hashtags"
    )
    parser.add_argument(
        "--split_emojis", type=bool, default=False,
        help="whether to split sequence of emojis by whitespace"
    )    

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