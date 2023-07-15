# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


######1
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os

#set path
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'


#read data file
ghg = pd.read_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed


#explore data structure
ghg.columns
ghg.dtypes
ghg.describe()
ghg['gdp'].value_counts(dropna = False).sort_index()




######2
#correlation matrix with all variables except strings and years
ghg_corr = ghg.drop(columns=['Unnamed: 0', 'country', 'year', 'iso_code'])
ghg_corr.corr()
plt.matshow(ghg_corr.corr()) #overwhelming amount of variables using the entire dataset


#to make correlation more readable, I will address a specific question.
#question 2: How do GHG emissions vary by source/what percentage of GHG emissions come from different sources?
ghg_sub = ghg[['cement_co2', 'co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'luc_co2', 'oil_co2', 
               'other_industry_co2', 'trade_co2', 'methane', 'nitrous_oxide']]




######3
#create correlation heatmap using matplotlib
plt.matshow(ghg_sub.corr())

#current dir
cwd = os.getcwd()
cwd

#add labels, legend, and change size of heatmap
plot = plt.figure(figsize=(8,8))
plt.matshow(ghg_sub.corr(), fignum=plot.number) #changes type of plot
plt.xticks(range(ghg_sub.shape[1]), ghg_sub.columns, fontsize=14, rotation=90) #x axis labels
plt.yticks(range(ghg_sub.shape[1]), ghg_sub.columns, fontsize=14) #y axis labels
clr = plt.colorbar() #add legend in shape of color bar
clr.ax.tick_params(labelsize=14) #add font size of ticks
plt.title('Correlation Matrix', fontsize=14) #add title to plot


# Correlation heatmap in seaborn by applying a heatmap onto the correlation matrix above.
sns.heatmap(ghg_sub.corr(), annot = True)





######4
#create scatterplot for the two most positively associated variables: methane and nitrous_oxide 0.99.
sns.lmplot(x = 'methane', y = 'nitrous_oxide', data = ghg)
plt.xlabel('Methane (million tonnes CO2 equivalent)') 
plt.ylabel('Nitrous oxide (million tonnes CO2 equivalent)')


#create scatterplot for the two most negatively associated variables: cement_co2 and trade_co2 -0.39.
sns.lmplot(x = 'cement_co2', y = 'trade_co2', data = ghg)
plt.xlabel('Cement CO2 (million tonnes)') 
plt.ylabel('Trade CO2 (million tonnes)')






######5
#further refine variables for ones wanting to include in pair plot: compare total major ghgs co2, 
#methane, nitrous_oxide
ghg_sub2 = ghg[['co2', 'methane', 'nitrous_oxide', 'total_ghg']]


#I couldn't get my diagonal histogram plots to show with a pairplot, but this scatter matrix works well
from pandas.plotting import scatter_matrix
scatter_matrix(ghg_sub2, figsize = (10, 10))




######6
#measure variables for only the latest year in the record 2021
ghg_2021 = ghg[ghg['year'] == 2021]
ghg_2021.describe()
ghg_2021.isnull().sum()

#ghg_per_capita would be a good variable to determine categories for high, medium, and low emitting citizens
sns.histplot(ghg_2021['co2_per_capita'], bins = 20, kde = True)
plt.xlabel('CO2 per capita (tonnes)') 

#deriving co2_per_capita_cat column
ghg_2021.loc[ghg_2021['co2_per_capita'] > 10, 'co2_per_capita_cat'] = 'High emitter citizens'
ghg_2021.loc[(ghg_2021['co2_per_capita'] <= 10) & (ghg_2021['co2_per_capita'] > 5), 'co2_per_capita_cat'] = 'Medium emitter citizens' 
ghg_2021.loc[ghg_2021['co2_per_capita'] <= 5, 'co2_per_capita_cat'] = 'Low emitter citizens'
ghg_2021['co2_per_capita_cat'].value_counts(dropna = False)



#categorial plot using emitter category created above to find emissions by coal_co2, which is the most carbon
#intensive energy source
ax = sns.catplot(x="coal_co2", y="co2_per_capita", hue="co2_per_capita_cat", data=ghg_2021)
ax.set(xticks=(0,25,50,75,100,125,150))
ax.set_xticklabels(np.arange(0,175,25))
plt.xlabel('Coal CO2 (million tonnes)') 
plt.ylabel('CO2 per capita (tonnes)')