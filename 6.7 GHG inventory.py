# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:47:27 2023

@author: npirt
"""

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


#set path
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed
pd.set_option('display.float_format', '{:.2f}'.format)

#import dataset
ghg = pd.read_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))


#2019 most complete and recent year of records
ghg_2019 = ghg.loc[ghg['year'] == 2019]
ghg_2019 = ghg_2019.round(4)

#remove regions for country based analyses
ghg_countries = ghg_2019.drop([441,613,3051,3223,3495,8907,14104,14276,14548,14820,15050,15222,15494,20313,21634,
                               25083,24388,26568,26840,29888,33162,33434,33606,33878,34584,34755,35027,35199,36259,
                               42407,42679,42851,48600,50079], axis=0, inplace=True)



#slide 2
#can use world values to determine breakdown of ghg types, sources
ghg_2019_world = ghg_2019.loc[ghg_2019['country'] == 'World']


#remove na's
ghg_2019_world = ghg_2019_world.dropna(axis=1)



#total ghg by ghg type
type_ghg = ghg_2019_world[["co2", "methane", "nitrous_oxide"]]
ghg_2019_world = ghg_2019_world.T #transpose
ghg_2019_world.to_csv(os.path.join(path, 'Data', 'ghg_type_2019.csv'))




#co2 emissions by source
source_co2 = ghg_2019_world[["cement_co2", "coal_co2", "flaring_co2", "gas_co2", "luc_co2", "oil_co2",
                             "other_industry_co2", "trade_co2"]]

#trade_co2 is 0, remove from dataframe
source_co2 = source_co2.drop(columns = 'trade_co2')
source_co2 = source_co2.T #transpose
source_co2.to_csv(os.path.join(path, 'Data', 'co2_source_2019.csv'))




#highest and lowest emitters total ghg emissions
ghg_2019.sort_values(by=['total_ghg'], ascending = True)

#5 lowest emitter countries: Fiji, Niue, Tuvalu, Nauru, Cook Islands


ghg_total = ghg_2019.sort_values(by=['total_ghg'], ascending = False)

#5 highest emitter countries: China, US, India, Indonesia, Russia




#slide 3-4
#add birth rate and infant mortality rate to main spreadsheet
birth_rate_raw = pd.read_csv(os.path.join(path, 'Data', 'birth rate.csv'))
infant_mort_raw = pd.read_csv(os.path.join(path, 'Data', 'infant mortality rate.csv'))


#transpose mortality rates and years
birth_rate = birth_rate_raw.melt(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                                 var_name='year', value_name='fertility_rate')
infant_mort = infant_mort_raw.melt(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                                 var_name='year', value_name='infant_mortality')


#change dataframe column names to prepare for join
birth_rate.rename(columns = {'Country Name': 'country',
                           'Country Code': 'iso_code',
                           'Indicator Name': 'indicator_name',
                           'Indicator Code': 'indicator_code'}, inplace=True)

infant_mort.rename(columns = {'Country Name': 'country',
                           'Country Code': 'iso_code',
                           'Indicator Name': 'indicator_name',
                           'Indicator Code': 'indicator_code'}, inplace=True)

#convert year columns to string type to be able to merge dataframes
ghg['year'] = ghg['year'].astype(str)
birth_rate['year'] = birth_rate['year'].astype(str)
infant_mort['year'] = infant_mort['year'].astype(str)

#merge dataframes to include birth rate and infant mortality characteristics
df_merged_all = pd.merge(pd.merge(ghg, birth_rate, on=['country', 'iso_code', 'year']), 
                         infant_mort, on=['country', 'iso_code', 'year'])
df_merged_all.isnull().sum()

#remove missing values to prepare for cluster analysis
df_all_rmna = df_merged_all.dropna()

#remove categorical variables
df_all_rmna = df_all_rmna.drop(columns = ['country', 'iso_code', 'indicator_name_x', 'indicator_code_x',
                                          'indicator_name_y', 'indicator_code_y'])




#elbow technique
num_cl = range(1, 10) # Defines the range of potential clusters in the data.
kmeans = [KMeans(n_clusters=i) for i in num_cl] # Defines k-means clusters in the range assigned above.

score = [kmeans[i].fit(df_all_rmna).score(df_all_rmna) for i in range(len(kmeans))] # Creates a score that represents 
#a rate of variation for the given cluster option.

score


#plot elbow curve using Pylab
pl.plot(num_cl,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

#Based on the chart above, we can see that the score greatly increases between 2 and 3, then levels off after 3.
#Therefore, the optimal number of clusters will be 3.


#k means clustering
#create k means object, n_jobs argument removed in new versions of library
kmeans = KMeans(n_clusters = 3) 


#fit k means object to data
kmeans.fit(df_all_rmna)

df_all_rmna['clusters'] = kmeans.fit_predict(df_all_rmna)




#view number of values for clusters
df_all_rmna.head()
df_all_rmna['clusters'].value_counts()
#2: 903, 1: 43, 0: 172

#assign clusters colors
df_all_rmna.loc[df_all_rmna['clusters'] == 2, 'cluster'] = 'dark purple'
df_all_rmna.loc[df_all_rmna['clusters'] == 1, 'cluster'] = 'purple'
df_all_rmna.loc[df_all_rmna['clusters'] == 0, 'cluster'] = 'pink'

#plot emissions vs gdp
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=df_all_rmna['gdp'], y=df_all_rmna['total_ghg'], hue=kmeans.labels_, s=100) 
# Here, you're subsetting `X` for the x and y arguments to avoid using their labels. 
# `hue` takes the value of the attribute `kmeans.labels_`, which is the result of running the k-means algorithm.
# `s` represents the size of the points you want to see in the plot.
ax.grid(False) # This removes the grid from the background.
plt.xlabel('GDP (USD)') # Label x-axis.
plt.ylabel('Total GHG emissions (million tonnes CO2)') # Label y-axis.
plt.show()


#aggregate dataframe variables by cluster
df_all_rmna.groupby('cluster').agg({'population':['mean', 'median'], 
                         'gdp':['mean', 'median'], 
                         'co2':['mean', 'median'],
                         'methane':['mean', 'median'],
                         'nitrous_oxide':['mean', 'median'],
                          'total_ghg':['mean', 'median']})

#convert gdp and population to millions for better readability
df_all_rmna['gdp'] = df_all_rmna['gdp']/1000000
df_all_rmna['population'] = df_all_rmna['population']/1000000




#export with clusters column to make visualizations in tableau
df_all_rmna.to_csv(os.path.join(path, 'Data', 'all_characteristics_clusters.csv'))






#slide 5
#determine distribution of total ghg variable
ghg.total_ghg.describe()

# =============================================================================
# count    6149.00
# mean      771.49
# std      3553.43
# min      -186.55
# 25%         8.44
# 50%        38.05
# 75%       151.15
# max     49758.23
# Name: total_ghg, dtype: float64
# =============================================================================

#separate countries into low, medium, and high emitters based on data distribution
#< 25%/8.44: low, 25-75% medium, > 75% high

#deriving ghg_emitter_category column
ghg.loc[ghg['total_ghg'] > 151.15, 'ghg_emitter_category'] = 'High emitter country'
ghg.loc[(ghg['total_ghg'] <= 151.15) & (ghg['total_ghg'] >= 8.44), 
             'ghg_emitter_category'] = 'Medium emitter country' 
ghg.loc[ghg['total_ghg'] < 8.44, 'ghg_emitter_category'] = 'Low emitter country'

ghg['ghg_emitter_category'].value_counts(dropna = False)

#save data file
ghg.to_csv(os.path.join(path, 'Data', 'ghg_emitter_cat.csv'))




