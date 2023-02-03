import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


#outlier function needed

def clean_wine_data(df):
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    return df



def train_val_test(df, stratify = None):
    
    seed = 7
    
    ''' This function is a general function to split our data into our train, validate, and test datasets. We put in a dataframe
    to then return us the datasets of train, validate and test.'''
    
    train, test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = None)
    
    validate, test = train_test_split(test, test_size = 0.5, random_state = seed, stratify = None)
    
    return train, validate, test
 kh