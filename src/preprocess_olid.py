"""
    Module for pre-processing OLID dataset (Task A)
"""

import argparse

import pandas as pd

from sys import argv

# define constants
OFF = "OFF"
NOT = "NOT"

# define types
Example = {"content":str, "label":int}

def preprocess(file: str, lang: str="english") -> list:
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
        "--out_file", default=None, help="path to output file")
    
    args = parser.parse_args(argv[1:])
    file = args.file
    lang = args.language
    
    if args.out_file == None:
        out_file = f"preprocessed_{file}"
    else:
        out_file = args.out_file
    
    data = preprocess(file=file, lang=lang)
    write_file(data, out_file)