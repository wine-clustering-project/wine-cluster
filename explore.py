import pandas as pd
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def barplot(df, f1, f2):
    '''
    This function is to create a barplot using a dataframe and 2 features as f1 and f2
    '''
    sns.set(style="white", rc={"grid.linewidth": 0.0})
    ax = sns.barplot(data = df, x = f1, y= f2, hue= 'type', hue_order = ['white', 'red'],
                color = 'firebrick', ec = 'black', alpha = .8)
    
    plt.legend(loc='center', bbox_to_anchor=(0.5, -.2), ncol=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    plt.title(f'Quality Score Compared To {f1.title()}')
    
    return plt.show()


def barplot2(df, f1, f2):
    '''
    This function is to create a barplot using a dataframe and 2 features as f1 and f2
    '''
    
    sns.set(style="white", rc={"grid.linewidth": 0.0})
    
    sns.barplot(data = df, x = f1, y= f2, color = 'firebrick', alpha = .8, ec = 'black',)
    
    plt.title(f'Quality Score Compared To {f2.title()}')
    
    return plt.show()


def ph_quality(train):
    '''
    This function is to create a barplot using a train dataset
    '''
    
    sns.set(style="white", rc={"grid.linewidth": 0.0})   
    sns.barplot(data = train, x = 'quality', y= 'ph', color = 'firebrick', hue= 'type',alpha = .8, ec = 'black',)
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
    '''
    This function is creating unscaled and scaled clusters and adding columns to the dataset
    '''
    
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
    '''
    this functions creates a relplot of the clusters
    '''
    
    sns.set(style = "whitegrid")
    
    X = clustering(df, f1, f2)
    
    sns.relplot(data = X, x = f1, y = f2, hue = 'scaled_clusters')
    
    plt.title('Clusters')
    
    return plt.show() 


def best_cluster(df, f1, f2):
    '''
    This function makes a graph to show the most optimal cluster number
    '''
    
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
    
    plt.title('The Best Cluster')
    
    return plt.show()


def qual_den(train):
    '''
    A function to run a 2 sample test and get the results
    '''
    
    alpha =0.05
    den= train.density
    qua= train.quality
    t, p = stats.ttest_ind(den, qua, equal_var=False)
    print("Are we able to reject the Null Hypothesis?:", p < alpha)
    print(f't = {t:.3f}')
    print(f'p-value = {p}')