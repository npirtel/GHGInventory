# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:17:57 2023

@author: npirt
"""

#import libraries
import pandas as pd
import numpy as np
import os

#set path
path = r'D:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

#import data
ghg = pd.read_csv(os.path.join(path, 'GHG data.csv'))


pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed

ghg.head()


#shorten column names to be more concise yet still descriptive
ghg.rename(columns = {'co2_including_luc': 'co2_w_luc',
                       'co2_including_luc_growth_abs': 'co2_w_luc_growth_abs',
                       'co2_including_luc_growth_prct': 'co2_w_luc_growth_prct',
                       'co2_including_luc_per_capita': 'co2_w_luc_per_capita',
                       'co2_including_luc_per_gdp': 'co2_w_luc_per_gdp',
                       'co2_including_luc_per_unit_energy': 'co2_w_luc_per_unit_energy',
                       'cumulative_co2_including_luc': 'cumulative_co2_w_luc',
                       'ghg_excluding_lucf_per_capita': 'ghg_wo_lucf_per_capita',
                       'land_use_change_co2': 'luc_co2',
                       'land_use_change_co2_per_capita': 'luc_co2_per_capita',
                       'share_global_co2_including_luc': 'share_global_co2_w_luc',
                       'share_global_cumulative_co2_including_luc': 'share_global_cumulative_co2_w_luc',
                       'share_of_temperature_change_from_ghg': 'share_of_temp_change_from_ghg',
                       'temperature_change_from_ch4': 'temp_change_from_ch4',
                       'temperature_change_from_co2': 'temp_change_from_co2',
                       'temperature_change_from_ghg': 'temp_change_from_ghg',
                       'temperature_change_from_n2o': 'temp_change_from_n2o',
                       'total_ghg_excluding_lucf': 'total_ghg_wo_lucf'}, inplace=True)



#data consistency checks
#locate rows with missing values
ghg.shape
ghg.isnull().sum()

#removing rows with NAs
ghg_rmna = ghg.dropna() 
ghg_rmna.shape

##Removing NAs from all columns would remove 98% of data, so none removed for now.
##Most rows have at least 1 column with no values and no columns have all missing values.


#find rows that are duplicates
ghg_dup = ghg[ghg.duplicated()]

#No duplicates found.


#check if any columns have mixed data
for col in ghg.columns.tolist():
  weird = (ghg[[col]].applymap(type) != ghg[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (ghg[weird]) > 0:
    print (col)

#No columns with mixed data.

#value counts for string columns to check for formatting issues and spelling.
ghg['iso_code'].value_counts(dropna = False).sort_index()
set = set(ghg['iso_code'])
num_values = len(set)
print(num_values) #232 unique values

ghg['country'].value_counts(dropna = False).sort_index()
set2 = set(ghg['country'])
num_values2 = len(set2)
print(num_values2) #278 unique values


#spread of numeric variables
ghg.describe()

#There are negative values included in many of the columns, but with
#ghg tracking, there can be negative CO2 change (decrease in CO2) and also
#very large increases, especially for countries that developed rapidly.



#data wrangling
#no columns deleted

#all column names consistent

ghg.dtypes
#country and iso_code columns string values, all other columns numeric: year
#integer and all others float type

#save as csv file
ghg.to_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))
