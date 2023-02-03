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
 
    
def the_split(df, stratify= None):
<<<<<<< HEAD
    
=======
        
    """ This functions is used to split the data into 3 different datasets: train, validate(val), and test.
        It then then returns the seperate datasets and prints the shape for each of them.
    """
>>>>>>> 7027c1f222fbe4ba5f1e7643e81ccf12cabe28c4
        
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test= train_test_split(df, test_size= .2, random_state= 7)
    train, val= train_test_split(train_validate, test_size= .3, random_state= 7)
    
    print(f'Train shape: {train.shape}\n' )
    
    print(f'Validate shape: {val.shape}\n' )
    
    print(f'Test shape: {test.shape}')
    
    return train, val, test