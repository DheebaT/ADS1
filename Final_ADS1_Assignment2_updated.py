# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:29:38 2022

@author: Dheeba
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns 
import matplotlib.pyplot as plt


# defining a function to read data
def readFile_Pop(fname):
    data = pd.read_csv(fname)
    return data

# mentioned the path of the data file 
# Used this data from world bank to show the popuplation growth in world over the years
Population_data =r"C:\Users\Dheeba\.spyder-py3\Data.csv"

# returning the function to a variable
df_data = readFile_Pop(Population_data)


#checking whether there are nay missing values in a data set
df_data.isnull()

#checking the number of missing values in each column
df_data.isna().sum()

#counting number of missing values in the dataset
print(df_data.isnull().values.sum())

# creating a data frame to select few countries and years
df1 = df_data[(df_data['Country Name'] == 'United Kingdom')
              |(df_data['Country Name'] == 'India')
                |(df_data['Country Name'] == 'United States')
                  |(df_data['Country Name'] == 'Netherlands')
                    |(df_data['Country Name'] == 'Germany')
                    |(df_data['Country Name'] == 'France')
                    |(df_data['Country Name'] == 'Japan')
                    |(df_data['Country Name'] == 'Malaysia')
                    |(df_data['Country Name'] == 'New Zealand')
                    |(df_data['Country Name'] == 'Norway')]


 
df1 = df1[['Country Name','1980','1990','2000','2010','2021']]
 # grouping the countries by mean
df1 = df1.groupby('Country Name').agg('mean')

print(df1)

#bar plot

def Bar_plot(fname):
    #selecting first 10 countries and storing it in a variable Countries
    Countries = fname['Country Name'].head(10).values
    x = np.arange(len(Countries))
    w = 0.3
    plt.figure(dpi=144, figsize = (20,10))
    plt.bar(x-w, fname['1980'].head(10), width = w, label = '1980')
    plt.bar(x, fname['1990'].head(10), width = w, label = '1990')
    plt.bar(x+w, fname['2000'].head(10), width = w, label = '2000')
    plt.xticks(x, Countries)
    plt.title("Population percentage annually")
    plt.xlabel("Countries")
    plt.ylabel("Percentage")
    plt.legend()
    plt.show()
    plt.savefig('barplot.png')

def Line_plot():
          
    #transposing years into one column for plotting line graph
    df2 = df1.transpose()
   
    #storing the transposed countries in a variable
    countries = df2.columns

    #line plot
    plt.figure(figsize = (20,10))
    plt.plot(df2, label = countries)
    plt.legend(loc='upper right')
    plt.title('Population growth annually')
    plt.xlabel('Years')
    plt.ylabel('Population percentage')
    plt.show()
    plt.savefig('lineplot.png')



# statistics for selected countries and years

def stats_func(var):
    print("Average: ", np.average(var))
    print("Std. deviations:", np.std(var))
    print("Skewness: ", stats.skew(var))
    print("Kurtosis: ", stats.kurtosis(var))
    return

print()
print("1980")
y1 = stats_func(df1["1980"])
print()
print("1990")
y2 = stats_func(df1["1990"])
print()
print("2000")
y3 = stats_func(df1["2000"])
print()
print("2010")
y4 = stats_func(df1["2010"])
print()
print("2021")
y5 = stats_func(df1["2021"])


# normalising statistics for the selected countries and years
def stat_norm(var):
    aver = np.average(var)
    stdev = np.std(var)
    var = (var-aver) / stdev
    return var


print()
print("1980")
y1 = stat_norm(df1["1980"])
stats_func(y1)

print()
print("1990")
y2 = stat_norm(df1["1990"])
stats_func(y2)

print()
print("2000")
y3 = stat_norm(df1["2000"])
stats_func(y3)

print()
print("2010")
y4 = stat_norm(df1["2010"])
stats_func(y4)

print()
print("2021")
y5 = stat_norm(df1["2021"])
stats_func(y5)


# subplotting histogram to show the normalised distribution
def Hist_stat():
    plt.figure(figsize =(30,10))
    plt.subplot(2, 3, 1)
    plt.hist(y1, bins=50, range=(-4,4.0), label = "1980")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.hist(y2, bins=50, range=(-4,4.0), label = "1990")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.hist(y3, bins=50, range=(-4,4.0), label = "2000")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.hist(y4, bins=50, range=(-4,4.0), label = "2010")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.hist(y5, bins=50, range=(-4,4.0), label = "2021")
    plt.legend()
    plt.show()
    plt.savefig('Histplot.png')


#correlation from the data frame created initially with few countries and years
def corr_coef():
    data_corr = df1.corr(method='pearson')
    print(data_corr)
    plt.figure(figsize =(16,10))
    sns.heatmap(data_corr, annot = True)
    plt.xticks(font=10)
    plt.yticks(font=10)
    plt.show()
    plt.savefig('Heatmap.png')
    
# calling the defined functions
Bar_plot(df_data)
Line_plot()
Hist_stat()
corr_coef()    

