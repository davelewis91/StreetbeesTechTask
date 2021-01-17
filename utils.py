"""
Useful utility functions for loading the data,
doing some preprocessing as needed, as well as
useful type declarations
"""
import re
from typing import NewType, List, Tuple, Dict

import pandas as pd

PDDataFrame = NewType('PDDataFrame', pd.DataFrame)

def make_class_map(class_file: str) -> Dict[int, str]:
    """Load the classes file and return the class names
    mapped to ints in a dict.

    Parameters
    ----------
    class_file: str
        Filepath to the classes file
    
    Returns
    -------
    classmap: dict
        Dictionary containing int-to-class name map
    """
    classmap = {}
    with open(class_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            classmap[i+1] = line
    return classmap

def preprocess_data(data: PDDataFrame) -> PDDataFrame:
    """Take the loaded text data and perform some preprocessing

    Parameters
    ----------
    data: pandas.DataFrame
        Input data to be preprocessed
    
    Returns
    -------
    data: pandas.DataFrame
        The preprocessed data
    """
    # remove punctuation (except - and ! which might be useful)
    data['Name'] = data['Name'].apply(
        lambda x: re.sub(r'[^\d\w -!.]', '', x)
    )
    # turn to lowercase
    data['Name'] = data['Name'].str.lower()
    return data

def load_data(path: str) -> PDDataFrame:
    """Load text data from a filepath and perform any
    preprocessing steps required

    Parameters
    ----------
    path: str
        Filepath to data file
    
    Returns
    -------
    data: pandas.DataFrame
        Loaded and preprocessed data
    """
    data = pd.read_csv(path)
    data = preprocess_data(data)
    return data