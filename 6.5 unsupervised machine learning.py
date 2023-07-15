# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 11:53:19 2023

@author: npirt
"""

######1
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.cluster import KMeans
import pylab as pl




######2
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed


#data has already been cleaned as described in previous exercises
ghg_clean = pd.read_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))

#remove missing values
ghg_rmna = ghg_clean.dropna()

#remove categorical variables
ghg_rmna = ghg_rmna.drop(columns = ['country', 'iso_code'])





######3
#elbow technique
num_cl = range(1, 10) # Defines the range of potential clusters in the data.
kmeans = [KMeans(n_clusters=i) for i in num_cl] # Defines k-means clusters in the range assigned above.

score = [kmeans[i].fit(ghg_rmna).score(ghg_rmna) for i in range(len(kmeans))] # Creates a score that represents 
#a rate of variation for the given cluster option.

score


#plot elbow curve using Pylab
pl.plot(num_cl,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()



######4
#Based on the chart above, we can see that the score greatly increases between 2 and 3, then levels off after 3.
#Therefore, the optimal number of clusters will be 3.





######5
#k means clustering
#create k means object, n_jobs argument removed in new versions of library
kmeans = KMeans(n_clusters = 3) 


#fit k means object to data
kmeans.fit(ghg_rmna)

ghg_rmna['clusters'] = kmeans.fit_predict(ghg_rmna)




######6
ghg_rmna.head()
ghg_rmna['clusters'].value_counts()
#0: 987, 1: 43, 2: 204





######7
#total ghg emissions vs gdp
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=ghg_rmna['gdp'], y=ghg_rmna['total_ghg'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.
ax.grid(False) # This removes the grid from the background.
plt.xlabel('GDP (USD)') # Label x-axis.
plt.ylabel('Total GHG emissions (million tonnes CO2)') # Label y-axis.
plt.show()



#total ghg emissions vs population
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=ghg_rmna['population'], y=ghg_rmna['total_ghg'], hue=kmeans.labels_, s=100) 
ax.grid(False) 
plt.xlabel('Population') 
plt.ylabel('Total GHG emissions (million tonnes CO2)')
plt.show()



#total ghg emissions vs year
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=ghg_rmna['year'], y=ghg_rmna['total_ghg'], hue=kmeans.labels_, s=100) 
ax.grid(False) 
plt.xlabel('Year')
plt.ylabel('Total GHG emissions (million tonnes CO2)')
plt.show()



#methane vs co2
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=ghg_rmna['co2'], y=ghg_rmna['methane'], hue=kmeans.labels_, s=100) 
ax.grid(False) 
plt.xlabel('CO2 emissions (million tonnes)') 
plt.ylabel('Methane emissions (million tonnes CO2 equivalent)') 
plt.show()




######8
#The clusters make sense because they seem to be separating the dataset into low, medium, and high variables
#generally in terms of emissions. Other variables such as gdp seem to follow this trend while other do not seem 
#to be as clear cut, such as population vs emissions, but in general these clusters are based on the range of 
#lower vs higher level of various variables.




######9
#assign cluster numbers colors
ghg_rmna.loc[ghg_rmna['clusters'] == 2, 'cluster'] = 'dark purple'
ghg_rmna.loc[ghg_rmna['clusters'] == 1, 'cluster'] = 'purple'
ghg_rmna.loc[ghg_rmna['clusters'] == 0, 'cluster'] = 'pink'


#aggregate dataframe variables by cluster
ghg_rmna.groupby('cluster').agg({'population':['mean', 'median'], 
                         'gdp':['mean', 'median'], 
                         'co2':['mean', 'median'],
                         'methane':['mean', 'median'],
                         'nitrous_oxide':['mean', 'median'],
                          'total_ghg':['mean', 'median']})

#Based on these aggregations, the different colored clusters do seem to be separated numerically of the variables
#listed and likely the others as well that are derivations of these general variables. The purple cluster/1 has
#the highest means and medians, followed by dark purple/2 and then pink/0 with the lowest mean and median values.





######10
#I can predict that these clusters are showing the low, medium, and high emitter countries and therefore
#will be useful for seeing how variables such as GDP and population vary with GHG emissions, which I visualized
#in question 7. Although I had to remove many countries in my analysis since no NA values were allowed in the
#k-means clustering, it was clear based on the analysis that these variables have a positive relationship
#that provides some evidence for my hypotheses and will also be helpful for answering some of my research
#questions by showing how emissions are changing over time.



