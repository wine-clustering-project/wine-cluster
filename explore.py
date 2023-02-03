import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def barplot(df, f1, f2):
    
    sns.set(style="white", rc={"grid.linewidth": 0.0})
    
    sns.barplot(data = df, x = f1, y= f2, color = 'dodgerblue', alpha = .8, ec = 'black',)
    
    plt.title(f'Quality Score Compared To {f1.title()}')
    
    return plt.show()

def barplot2(df, f1, f2):
    
    sns.set(style="white", rc={"grid.linewidth": 0.0})
    
    sns.barplot(data = df, x = f1, y= f2, color = 'dodgerblue', alpha = .8, ec = 'black',)
    
    plt.title(f'Quality Score Compared To {f2.title()}')
    
    return plt.show()


def ph_quality():
    sns.set(style="white", rc={"grid.linewidth": 0.0})   
    sns.barplot(data = train, x = 'quality', y= 'ph', color = 'firebrick', hue= 'type',alpha = 1, ec = 'black',)
    plt.legend(loc= 'lower right')
    plt.xlabel('Quality of The Wine')
    plt.ylabel('Ph Level of The Wine')
    plt.title('PH of Each Wine Type')


def chi_test(feature, df):
    '''get result of chi-square for a feature to quality'''
    
    𝜶 = .05

    observed = pd.crosstab(df[feature], df.quality)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    if p < 𝜶:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")

    print(f'chi² = {chi2:.3f}')
    print(f'p = {p:.3}')
    

def clustering(train, f1, f2):
    
    seed = 22
    
    X = train[[f1, f2]]
    
    kmeans = KMeans(n_clusters = 4, random_state= seed)
    kmeans.fit(X)
    kmeans.predict(X)

    X['unscaled_clusters'] = kmeans.predict(X)
    
    mm_scaler = MinMaxScaler()
    X[[f1, f2]] = mm_scaler.fit_transform(X[[f1, f2]])
    
    kmeans_scale = KMeans(n_clusters = 4, random_state = 22)
    kmeans_scale.fit(X[[f1, f2]])
    kmeans_scale.predict(X[[f1, f2]])
    
    X['scaled_clusters'] = kmeans_scale.predict(X[[f1, f2]])
    
    return X    
   

def cluster_relplot(df, f1, f2):
    
    sns.set(style = "whitegrid")
    
    X = clustering(df, f1, f2)
    
    sns.relplot(data = X, x = f1, y = f2, hue = 'scaled_clusters')
    
    return plt.show() 


def best_cluster(df, f1, f2):
    
    X = clustering(df, f1, f2)
    
    inertia = []
    seed = 22 

    for n in range(1,11):

        kmeans = KMeans(n_clusters = n, random_state = seed)

        kmeans.fit(X[[f1, f2]])

        inertia.append(kmeans.inertia_)
        
        
    results_df = pd.DataFrame({'n_clusters': list(range(1,11)),
                               'inertia': inertia})   
    
    sns.set_style("whitegrid")
    sns.relplot(data = results_df, x='n_clusters', y = 'inertia', kind = 'line')
    plt.xticks(np.arange(0, 11, step=1))
    point = (3, 107) # specify the x and y values of the point to annotate
    plt.annotate("optimal cluster", xy=point, xytext=(5, 125), 
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    return plt.show()