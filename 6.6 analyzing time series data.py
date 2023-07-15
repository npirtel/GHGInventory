# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:41:49 2023

@author: npirt
"""

######2
#import libraries
#import quandl #my dataset already includes a time series component so it won't be necessary to upload data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm # Using .api imports the public access version of statsmodels, which is a library 
#that handles statistical models.
import os
import warnings # This is a library that handles warnings.

warnings.filterwarnings("ignore") # Disable deprecation warnings that could indicate, for instance, a suspended 
#library or feature. These are more relevant to developers and very seldom to analysts.

plt.style.use('fivethirtyeight') # This is a styling option for how your plots will appear. More examples here:
# https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
# https://matplotlib.org/3.1.0/gallery/style_sheets/fivethirtyeight.html


#set path
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed





######3
#I would like to look at the co2 emissions of China, now the largest emitter, over the years where there are records
#All years are relevant bc they capture China's rapid industrialization. So my dataset is subset by country but
#not by time records.

#import dataset
ghg = pd.read_csv(os.path.join(path, 'Data', 'ghg_data_cleaned.csv'))


#subset for China
china = ghg.loc[ghg['country'] == 'China']


#analyze only co2 column and clean subset
china_co2 = china[["country", "year", "co2"]]



#remove na's and 0 values
china_co2 = china_co2.dropna()
china_co2.isnull().sum() #no na's

#model decomposition only works for non 0 values
china_co2 = china_co2.loc[china_co2['co2'] > 0]


#check for duplicate values
dups = china_co2.duplicated() #dataframe has no columns/no duplicates

#remove country column as it is no longer needed
china_co2 = china_co2.drop(columns = ['country'])


#make date column to use datetime function and set date as index
from datetime import datetime

china_co2['month'] = [1] * 115
china_co2['day'] = [1] * 115

china_co2['date'] = pd.to_datetime(china_co2[['year', 'month', 'day']])

#Reset index to use the "date" column as a filter
china_co2['datetime'] = pd.to_datetime(china_co2['date']) # Create a datetime column from "date.""
china_co2 = china_co2.set_index('datetime') # Set the datetime as the index of the dataframe.
china_co2.drop(['date', 'year', 'month', 'day'], axis=1, inplace=True) # Drop the unneeded column.





######4
#plot dataset
#data has a clear upward trend starting in the 1960s as China begins to industrialize
plt.figure(figsize=(15,5), dpi=100)
plt.plot(china_co2)






######5
#decompose the time series using a multiplicative model, as co2 levels increase exponentially
decomposition = sm.tsa.seasonal_decompose(china_co2, model='multiplicative')


from pylab import rcParams # This will define a fixed size for all special charts.
rcParams['figure.figsize'] = 18, 7


#plot the separate components
decomposition.plot()
plt.show()


##The shape of the data and the trend show a clear exponential increase starting around halfway in the time series.
##The data appear to not have seasonality and residual of 1, which means the there is not unexplained noise.






######6
#Dickey-Fuller test to check for stationary data
from statsmodels.tsa.stattools import adfuller # Import the adfuller() function

def dickey_fuller(timeseries): # Define the function
    # Perform the Dickey-Fuller test:
    print ('Dickey-Fuller Stationarity test:')
    test = adfuller(timeseries, autolag='AIC')
    result = pd.Series(test[0:4], 
                       index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
    for key,value in test[4].items():
       result['Critical Value (%s)'%key] = value
    print (result)

# Apply the test using the function on the time series
dickey_fuller(china_co2['co2'])

# =============================================================================
# Dickey-Fuller Stationarity test:
# Test Statistic                   6.619398
# p-value                          1.000000
# Number of Lags Used             13.000000
# Number of Observations Used    101.000000
# Critical Value (1%)             -3.496818
# Critical Value (5%)             -2.890611
# Critical Value (10%)            -2.582277
# dtype: float64
# =============================================================================

##The test statistic is extremely positive and is the reason why the p-value is so high. This indicates that
##our data are definitely not stationary and I cannot reject the null hypothesis. In order to do so, the test
##statistic would have to be below -2.89.


# Check out a plot of autocorrelations
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Here, you import the autocorrelation and partial 
#correlation plots

plot_acf(china_co2)
##The non stationary data finding is supported by the various autocorrelations between lags of the data.





######7
#performing differencing to stationarize the data
data_diff = china_co2 - china_co2.shift(1) # The df.shift(1) function turns the observation to t-1, making the 
#whole thing t - (t -1)

data_diff.dropna(inplace = True) #drop na to rerun the dickey-fuller test


#check what differencing did to time-series curve
plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_diff)


#rerun test
dickey_fuller(data_diff)

# =============================================================================
# Dickey-Fuller Stationarity test:
# Test Statistic                   0.047656
# p-value                          0.962300
# Number of Lags Used             13.000000
# Number of Observations Used    100.000000
# Critical Value (1%)             -3.497501
# Critical Value (5%)             -2.890906
# Critical Value (10%)            -2.582435
# dtype: float64
# =============================================================================

plot_acf(data_diff)

##Based on the test results and the autocorrelations plot, the data are still not stationary but are closer.




######8
#differencing round 2
data_diff2 = data_diff - data_diff.shift(1) 

data_diff2.dropna(inplace = True) 


#check what differencing did to time-series curve
plt.figure(figsize=(15,5), dpi=100)
plt.plot(data_diff2)


#rerun test
dickey_fuller(data_diff2)


# =============================================================================
# Dickey-Fuller Stationarity test:
# Test Statistic                -5.984997e+00
# p-value                        1.800969e-07
# Number of Lags Used            1.300000e+01
# Number of Observations Used    9.900000e+01
# Critical Value (1%)           -3.498198e+00
# Critical Value (5%)           -2.891208e+00
# Critical Value (10%)          -2.582596e+00
# dtype: float64
# =============================================================================


plot_acf(data_diff2)


##Based on the results of the test and the plots, we can now conclude that the data are stationary.
##Next time, we can also try transforming the data logarithmically to try to reach stationary status.

