# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 06:33:34 2023

@author: Dheeba
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import clear_output
from scipy.optimize import curve_fit

df = pd.read_csv(r'C:\Users\Dheeba\.spyder-py3\CO2_Emissions.csv', skiprows = 4)

print(df.columns)
years = ['1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019']

df = df.dropna(subset = years)

df1 = df[years].copy()
print(df1)
#scaling the data
#df1 = (df - df.min()) / (df.max() - df.min()) * 9 + 1

print(df1.describe())


def random_centroids(df1, k):
    centroids = []
    for i in range(k):
        centroid = df1.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis = 1)

centroids = random_centroids(df1, 100)
print(centroids)
        

def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((df1 - x) ** 2).sum(axis = 1)))
    return distances.idxmin(axis = 1)


labels = get_labels(df1, centroids)
print(labels.value_counts())


def new_centroids(df1, labels, k):
    return df1.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


def plot_clusters(df1, labels, centroids, iteration):
    pca = PCA(n_components = 2)
    data_2d = pca.fit_transform(df1)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait = True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x = data_2d[:,0], y = data_2d[:,1], c = labels)
    plt.scatter(x = centroids_2d[:,0], y = centroids_2d[:,1])
    plt.show()

max_iterations = 100
k = 3

centroids = random_centroids(df1, k)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    
    labels = get_labels(df1, centroids)
    centroids = new_centroids(df1, labels, k)
    plot_clusters(df1, labels, centroids, iteration)
    iteration += 1

print(centroids)    


#Curve Fit

pca = PCA(n_components=2)

pca.fit(df1)

df_pca = pca.transform(df1)

print(df_pca)

plt.scatter(df_pca[:,0],df_pca[:,0])

plt.show()


from scipy.optimize import curve_fit


def func_logistic(x, a, b, c):
    return c / (1 + np.exp(-b*(x-a)))

popt, pcov = curve_fit(func_logistic, df_pca[:, 0], df_pca[:, 0])

plt.scatter(df_pca[:, 0], df_pca[:, 0], color='red')
x = np.linspace(df_pca[:, 0].min(), df_pca[:, 0].max(), 100)
y = func_logistic(x, *popt)
plt.plot(x, y, color='blue')
plt.show()
