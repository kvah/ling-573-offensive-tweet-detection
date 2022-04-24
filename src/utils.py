# Script for any useful shared util functions 

import pandas as pd

def df_from_indices(indices_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a subset of a pandas dataframe based on a filepath containing 
    a list of indices
    """
    with open(indices_path, 'r') as fp:
        ids = [int(line.rstrip()) for line in fp]
    return df.loc[ids]

