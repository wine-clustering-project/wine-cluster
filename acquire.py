import pandas as pd
import numpy as np
import os




def get_wine_data():
    
    '''
    This function is to get the wine dataset from a local csv file that is already combined from the 2 csvs that were 
    downloaded from data.world if not combine the 2 csvs and make return a df to our working notebook to be able to
    use the data and perform various tasks using the data
    ''' 
    
    if os.path.isfile('wines.csv'):
        
        return pd.read_csv('wines.csv')
    
    else:
        
        red_wine = pd.read_csv("red_wine.csv")
        white_wine = pd.read_csv("white_wine.csv")

        # Add a new column to each dataframe indicating the type of wine
        red_wine["type"] = "red"
        white_wine["type"] = "white"

        # Concatenate the two dataframes to create a single combined dataframe
        wines = pd.concat([red_wine, white_wine])

        # Save the combined dataframe to a new CSV file
        wines.to_csv("wines.csv", index=False)

        return wines  