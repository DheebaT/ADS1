# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:09:07 2023

@author: Dheeba
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import t


df = pd.read_csv(r'C:\Users\Dheeba\.spyder-py3\CO2_Emissions.csv', skiprows = 4)

#creating a new dataframe with a list of Years from the dataset
years = ['1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019']



#dropping any null values present in the selected dataframe
df = df.dropna(subset = years)

#copying the years into df
df = df[years].copy()
print(df)

#scaling the data
df1 = (df - df.min()) / (df.max() - df.min()) * 9 + 1

#describing each column from the dataframe
print(df1.describe())


# Initialising random centroids
def random_centroids(df1, k):
    centroids = []
    for i in range(k):
        #sample selects one value from each column of the DataFrame.
        centroid = df1.apply(lambda x: float(x.sample()))
        #appending randomly selected centroid to centroid
        centroids.append(centroid)
        #concatenating list of centroids along the columns axis
    return pd.concat(centroids, axis = 1)

centroids = random_centroids(df1, 100)
print(centroids)
        

#labelling each data point
def get_labels(df1, centroids):
    #finding the Euclidean distance between each row in the input dataframe and applying the lambda function to each column of the DataFrame
    distances = centroids.apply(lambda x: np.sqrt(((df1 - x) ** 2).sum(axis = 1)))
    #returning the index of the column with the minimum distance for each row 
    return distances.idxmin(axis = 1)

labels = get_labels(df1, centroids)
print(labels.value_counts())

#Updating new centroids
def new_centroids(df1, labels, k):
    return df1.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


#plotting the clusters
def plot_clusters(df1, labels, centroids, iteration):
    pca = PCA(n_components = 2)
    #using pca on the input data to reduce the dimensionality of the data to 2 dimensions
    data_2d = pca.fit_transform(df1)
    #transposing the centroids
    centroids_2d = pca.transform(centroids.T)
    #clearing the existing output
    clear_output(wait = True)
    plt.title(f'Iteration {iteration}')
    #visualising using scatter plot
    plt.scatter(x = data_2d[:,0], y = data_2d[:,1], c = labels)
    plt.scatter(x = centroids_2d[:,0], y = centroids_2d[:,1], color='red')
    plt.show()

max_iterations = 100
k = 3

centroids = random_centroids(df1, k)
#used to check if the centroids have changed after each iteration
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    #assigning labels to each data point based on the closest centroid
    labels = get_labels(df1, centroids)
    #updating the centroids to the mean of the points assigned to each cluster
    centroids = new_centroids(df1, labels, k)
    #calling the plot cluster function to visualise
    plot_clusters(df1, labels, centroids, iteration)
    iteration += 1

print(centroids)    




# creating PCA with 2 components
pca = PCA(n_components=2)

# Fitting the PCA model to the dataset
pca.fit(df1)

# transforming the dataset
df1_pca = pca.transform(df1)

print(df1_pca)

#plotting the transformed data set
plt.scatter(df1_pca[:,0],df1_pca[:,1])

plt.show()


#Curve Fit
#Defining a function for polynomial curve fitting
def func_poly(x, a, b, c):
    return a*x**2 + b*x + c


# Fitting the function to the data 
popt, pcov = curve_fit(func_poly, df1_pca[:, 0], df1_pca[:, 1])

# Calculating the std dev
perr = np.sqrt(np.diag(pcov))

# Generating the x values for the best fit curve and the confidence ranges
x_best = np.linspace(df1_pca[:, 0].min(), df1_pca[:, 0].max(), 100)
y_best = func_poly(x_best, *popt)

y_conf = []
for i in range(3):
    y_conf.append(func_poly(x_best, popt[0] + perr[0]*norm.ppf(0.95), popt[1] + perr[1]*norm.ppf(0.95), popt[2] + perr[2]*norm.ppf(0.95)))

# Plotting the data, the best fit curve, and the confidence ranges
plt.scatter(df1_pca[:, 0], df1_pca[:, 1], color='yellow')
plt.plot(x_best, y_best, color='blue', label='Best Fit')
for i in range(3):
    plt.fill_between(x_best, y_conf[i], y2=y_best, color='blue', alpha=0.1)

plt.legend()
plt.show()

#metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# finding the silhouette_score
SilhouetteScore = silhouette_score(df1, labels)
print("Silhouette Score: ", SilhouetteScore)

# finding the davies_bouldin_score
Davies_score = davies_bouldin_score(df1, labels)
print("Davies-Bouldin Index: ", Davies_score)

# finding the calinski_harabasz_score
Calinski_score = calinski_harabasz_score(df1, labels)
print("Calinski-Harabasz index: ", Calinski_score)

# Choosing the figure size for the scatter plots
fig, ax = plt.subplots(figsize=(15, 8))

# scatter plotting for the year 1990
ax.scatter(df1.index, df1['1990'], c=labels, cmap='Dark2', label='1990')
# scatter plotting for the year 2000
ax.scatter(df1.index, df1['2000'], c=labels, cmap='Reds', label='2000')
# scatter plotting for the year 2010
ax.scatter(df1.index, df1['2010'], c=labels, cmap='Greens', label='2010')
# scatter plotting for the year 2015
ax.scatter(df1.index, df1['2015'], c=labels, cmap='Oranges', label='2015')
# scatter plotting for the year 2019
ax.scatter(df1.index, df1['2019'], c=labels, cmap='GnBu', label='2019')

plt.legend()
plt.title('CO2 Emissions')
plt.show()



#Err_ranges
# given value
alpha = 0.05
# creating a new dataframe
df2 = len(df1) - 1
# finding mean value
mean = np.mean(df1)
# finding standard deviation
stdev = np.std(df1)
# calculating the lower and upper limits 
ci = t.interval(1 - alpha, df2, loc=mean, scale=stdev)
print("Lower limit: ",ci[0])
print("Upper limit: ",ci[1])
