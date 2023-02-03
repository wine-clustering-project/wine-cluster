import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

import explore as e

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt 




def combined_df(df, f1, f2):
    '''
    This function calls another function in explore.py and merges a column to the original dataset
    '''
    
    X = e.clustering(df, f1, f2)
    
    scaled_clusters = X['scaled_clusters']
    df = pd.merge(df, scaled_clusters, left_index=True, right_index=True)
    
    return df


def mvp_scaled_data(train, validate, test, return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    columns_scale = train.iloc[:, :11]
    columns_to_scale = columns_scale.columns
    
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mms.transform(validate[columns_to_scale]), 
                                                     columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
    
def splitting_subsets(train, train_scaled, validate_scaled, test_scaled):
    '''
    This function splits our train, validate, and test scaled datasets into X/y train,
    validate, and test subsets
    '''
    
    
    X_train = train_scaled.drop(columns = ['quality'])
    X_train = pd.get_dummies(X_train, columns = ['type', 'scaled_clusters'])
    y_train = train_scaled['quality']


    X_validate = validate_scaled.drop(columns = ['quality'])
    X_validate = pd.get_dummies(X_validate, columns = ['type', 'scaled_clusters'])
    y_validate = validate_scaled['quality']


    X_test = test_scaled.drop(columns = ['quality'])
    X_test = pd.get_dummies(X_test, columns = ['type', 'scaled_clusters'])
    y_test = test_scaled['quality']

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def baseline(y_train):
    '''
    This function takes in y_train to calculate the baseline rmse
    '''
    
    preds_df = pd.DataFrame({'actual': y_train})
    
    preds_df['baseline'] = y_train.mean()
    
    baseline_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.baseline))

    return baseline_rmse


def linear_model(X_train, y_train):
    '''
    This function makes a linear regression model, fits, and predicts the output values.
    Giving us a dataframe of predicted linear and actual values
    '''
    
     '''
        This function is used to get a liner model. The results will be used on the validate dataset.
     '''
    
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_preds = lm.predict(X_train)
    
    preds_df = pd.DataFrame({'actual': y_train,'lm_preds': lm_preds})
    
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    
    df = pd.DataFrame({'model': 'linear', 'linear_rmse': lm_rmse},index=['0']) 
                      
    return df


def lasso_lars(X_train, y_train):
    
     '''
        This function is used to run a for loop on lasso lars. We will use the best preforming model,
        and use it on the validate datasets.
     '''
    
    metrics = []

    for i in np.arange(0.05, 1, .05):
    
        lasso = LassoLars(alpha = i )
    
        lasso.fit(X_train, y_train)
    
        lasso_preds = lasso.predict(X_train)
        
        preds_df = pd.DataFrame({'actual': y_train})
    
        preds_df['lasso_preds'] = lasso_preds

        lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    
        output = {
                'alpha': i,
                'lasso_rmse': lasso_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('lasso_rmse')


def tweedie_models(X_train, y_train):
    
     '''
        This function is used to run a for loop on tweedie model. We will use the best preforming model,
        and use it on the validate datasets.
    '''
    
    metrics = []

    for i in range(0, 4, 1):
    
        tweedie = TweedieRegressor(power = i)
    
        tweedie.fit(X_train, y_train)
    
        tweedie_preds = tweedie.predict(X_train)
        
        preds_df = pd.DataFrame({'actual': y_train})
    
        preds_df['tweedie_preds'] = tweedie_preds
    
        tweedie_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.tweedie_preds))
    
        output = {
                'power': i,
                'tweedie_rmse': tweedie_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('tweedie_rmse') 


def linear_poly(X_train, y_train):
    
    '''
        This function is used to run a for loop on liner poly. We will use the best preforming model,
        and use it on the validate datasets.
    '''
    
    metrics = []

    for i in range(2,4):

        pf = PolynomialFeatures(degree = i)

        pf.fit(X_train, y_train)

        X_polynomial = pf.transform(X_train)

        lm2 = LinearRegression()

        lm2.fit(X_polynomial, y_train)
        
        preds_df = pd.DataFrame({'actual': y_train})

        preds_df['poly_preds'] = lm2.predict(X_polynomial)

        poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))

        output = {
                'degree': i,
                'poly_rmse': poly_rmse
                 }

        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('poly_rmse') 


def validate_models(X_train, y_train, X_validate, y_validate):
    '''
            This model is used to test our models on the validate datasets and then return the results. 
            These results will then be used to find our best model.
    '''
       
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_val = lm.predict(X_validate)
    
    val_preds_df = pd.DataFrame({'actual_val': y_validate})
    
    val_preds_df['lm_preds'] = lm_val

    lm_rmse_val = sqrt(mean_squared_error(val_preds_df['actual_val'], val_preds_df['lm_preds']))

    #tweedie model
    
    tweedie = TweedieRegressor(power = 1)
    
    tweedie.fit(X_train, y_train)
    
    tweedie_val = tweedie.predict(X_validate)
    
    val_preds_df['tweedie_preds'] = tweedie_val
    
    tweedie_rmse_val = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df.tweedie_preds))
    
    #polynomial model
    
    pf = PolynomialFeatures(degree = 2)
    
    pf.fit(X_train, y_train)
    
    X_train = pf.transform(X_train)
    X_validate = pf.transform(X_validate)
    
    lm2 = LinearRegression()
    
    lm2.fit(X_train, y_train)
    
    val_preds_df['poly_vals'] = lm2.predict(X_validate)
    
    poly_validate_rmse = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['poly_vals']))

    #lasso_lars model
    
    lasso = LassoLars(alpha = .05 )
    
    lasso.fit(X_train, y_train)
    
    lasso_val = lasso.predict(X_validate)
    
    val_preds_df['lasso_preds'] = lasso_val

    lasso_rmse_val = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['lasso_preds']))
    
    
    return lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse


def test_model(X_train, y_train, X_test, y_test):
<<<<<<< HEAD
=======
    
    '''
        This function is used to test our best model and use it on the test datasets to get our final results.
    '''
    
    pf = PolynomialFeatures(degree = 3)

    pf.fit(X_train, y_train)
    X_train = pf.transform(X_train)

    X_test = pf.transform(X_test)
>>>>>>> 9bc436ed79aaf72329e27f77e901f3d7e6a36b5f

    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_preds = lm.predict(X_test)

    test_preds_df = pd.DataFrame({'actual_test': y_test})

    test_preds_df['linear_test'] = lm.predict(X_test)

    linear_test_rmse = sqrt(mean_squared_error(test_preds_df.actual_test, test_preds_df['linear_test']))
    
    return linear_test_rmse


def best_models(X_train, y_train, X_validate, y_validate):
    
    '''
        This function uses the train and validate datasets and returns the results of the best preforming model 
        for each algorithm. The results are returned as a dataframe.
    '''
    
    lm_rmse = linear_model(X_train, y_train).iloc[0,1]
    
    lasso_rmse = lasso_lars(X_train, y_train).iloc[0,1]
    
    tweedie_rmse = tweedie_models(X_train, y_train).iloc[0,1]
        
    poly_rmse = linear_poly(X_train, y_train).iloc[1,1]
    
    baseline_rmse = baseline(y_train)
    
    lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse = validate_models(X_train, y_train, X_validate, y_validate)
    
    df = pd.DataFrame({'model': ['linear', 'tweedie', 'lasso_lars','linear_poly', 'baseline'],
                      'train_rmse': [lm_rmse, tweedie_rmse, lasso_rmse, poly_rmse,  baseline_rmse],
                      'validate_rmse': [lm_rmse_val, tweedie_rmse_val, lasso_rmse_val, poly_validate_rmse, baseline_rmse]})
    
    df['difference'] = df['train_rmse'] - df['validate_rmse']
    
    return df.sort_values('difference').reset_index().drop(columns = ('index'))


def best_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    
    '''
        Takes in our datasets, gets the best preforming model, and runs it on the test datasets.
        Then it returns the results as a dataframe.
    '''
    df = best_models(X_train, y_train, X_validate, y_validate).iloc[1]
    
    df['test_rmse'] = test_model(X_train, y_train, X_test, y_test)
    
    df = pd.DataFrame(df).T
    
    df = df.drop(columns = ['difference'])

    return df