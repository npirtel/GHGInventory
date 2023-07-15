# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:23:09 2023

@author: npirt
"""

######2
#import libraries and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#set path
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed




######3
#the ghg dataset was cleaned in exercise 6.1, but the following steps were taken below:

#import data
ghg = pd.read_csv(os.path.join(path, 'Data', 'GHG data.csv'))


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




#the gdp data file was cleaned in exercise 6.3 as well
#drop na values
gdp_fill = pd.read_csv(os.path.join(path, 'Data', 'gdp_filled.csv'))

gdp_fill2021 = gdp_fill[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2021"]]
gdp_fill2021 = gdp_fill2021.dropna()
gdp_fill2021.isnull().sum()


#convert values to millions for better readability
gdp_fill2021['2021 millions'] = gdp_fill2021['2021']/1000000
gdp_fill2021.describe()


#duplicate values
dups = gdp_fill2021.duplicated() #dataframe has no columns/no duplicates


#check if any columns have mixed data
for col in gdp_fill2021.columns.tolist():
  weird = (gdp_fill2021[[col]].applymap(type) != gdp_fill2021[[col]].iloc[0].apply(type)).any(axis = 1)
  if len (gdp_fill2021[weird]) > 0:
    print (col)
#no mixed data





######4
#I already know that I would like to explore the relationship between GDP and CO2 emissions as continued from
#the previous exercise for the most recent year 2021.

#read in cleaned data sets
ghg_clean = pd.read_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))
gdp_clean = pd.read_csv(os.path.join(path, 'Data', 'gdp_filled.csv'))


#filter for year 2021 only 
ghg_allyr = ghg_clean[["country", "year", "co2"]]
ghg_2021 = ghg_allyr.loc[ghg_allyr['year'] == 2021]

gdp_2021 = gdp_clean[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2021"]]


#missing values for each dataset
ghg_2021.isnull().sum() #23
gdp_2021.isnull().sum() #3

#remove missing values
ghg_2021 = ghg_2021.dropna()
gdp_2021 = gdp_2021.dropna()

#change gdp_2021 column names to prepare for join/be consistent with other dataframe
gdp_2021.rename(columns = {'Country Name': 'country',
                           'Country Code': 'country_code',
                           'Indicator Name': 'indicator_name',
                           'Indicator Code': 'indicator_code'}, inplace=True)


#perform join
df_merged = ghg_2021.merge(gdp_2021, on = 'country', indicator='exists')

#drop indicator column
df_merged.drop(['exists'], axis=1, inplace=True)

#convert values to millions for better readability
df_merged['2021 millions'] = df_merged['2021']/1000000


#scatterplot to visualize the gdp and co2 relationship
sns.lmplot(x = '2021 millions', y = 'co2', data = df_merged)
plt.ylabel('CO2 (million tonnes)') 
plt.xlabel('GDP (million USD)')





######5
#hypothesis of gdp and co2 relationship
#The greater the country's gdp, the higher the co2 emissions.





######6
#reshape variables in NumPy arrays and put into separate objects
X = df_merged['2021 millions'].values.reshape(-1,1)
y = df_merged['co2'].values.reshape(-1,1)





######7
#split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)




######8 - regression analysis
#create regression object
regression = LinearRegression()


#fit regression object onto training set
regression.fit(X_train, y_train)


#predict y values from x - test set
y_predicted = regression.predict(X_test)


#training set
y_predicted_train = regression.predict(X_train)





######9
#plot to show regression line from model on test set
plot_test = plt
plot_test.scatter(X_test, y_test, color='gray', s = 15)
plot_test.plot(X_test, y_predicted, color='red', linewidth =3)
plot_test.title('2021 CO2 emissions vs GDP (Test set)')
plot_test.xlabel('GDP')
plot_test.ylabel('CO2 emissions')
plot_test.show()


#training set
plot_test = plt
plot_test.scatter(X_train, y_train, color='green', s = 15)
plot_test.plot(X_train, y_predicted_train, color='red', linewidth =3)
plot_test.title('2021 CO2 emissions vs GDP (Train set)')
plot_test.xlabel('GDP')
plot_test.ylabel('CO2 emissions')
plot_test.show()



######10
#Besides about 3 data points, the fit of the model seems to be good, although there is one outlier that is
#isgnificantly higher than the line of best fit, which I think will greatly reduce the R2 and increase the RMSE.
#I believe this country is China that has significantly higher emissions than the rest.





######11
#Create objects that contain the model summary statistics test set.
rmse = mean_squared_error(y_test, y_predicted) # This is the mean squared error
r2 = r2_score(y_test, y_predicted) # This is the R2 score. 

print('Slope:' ,regression.coef_) #0.00021
print('Mean squared error: ', rmse) #1,078,785.45
print('R2 score: ', r2) #0.52


#training set
rmse = mean_squared_error(y_train, y_predicted_train)
r2 = r2_score(y_train, y_predicted_train)

print('Slope:' ,regression.coef_) #0.00021
print('Mean squared error: ', rmse) #10,721
print('R2 score: ', r2) #0.95




######12
#create dataframe comparing actual and predicted y values.
data = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predicted.flatten()})
data.head(30)





######13
#Looking on how the model performed on the training set vs the test set, it did a lot worse on the test set.
#This means that the model overfit on the training set and was specific to that data set. This makes sense
#considering the few data points in the model and also how the model doesn't do as well with predicting outliers.
#To improve the testing set fit, it would help to remove the major outliers, but the countries that these points
#represent, such as China, the US, and India are all important to include in the analyses as they are the major
#CO2 emitters in the world, despite these points biasing the data patterns. This analysis would be better done on 
#a larger dataset, but the regression did support my hypothesis.

