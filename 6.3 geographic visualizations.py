# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:31:38 2023

@author: npirt
"""

######3
#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
import folium
#!pip install folium #if folium installation doesn't work from command prompt
import json


#set path
path = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project'

pd.set_option('display.max_columns', None) #display all columns
pd.options.display.max_rows = None #no limit to rows displayed





######4
# Import ".json" file for the countries
country_geo = r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project\\Data\\world-countries.json'

#see contents of json file
f = open(r'E:\\Data analyst course\\Data immersion\\Part 6 - Independent project\\Data\\world-countries.json',)
data = json.load(f)

for i in data['features']:
    print(i)
    




######5
#import gdp data file
gdp = pd.read_csv(os.path.join(path, 'Data', 'gdp.csv'))

#select only necessary columns - analyze gpd for 2021
gdp2021 = gdp[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2021"]]



#clean data

#remove regions that are not relevant for country based analysis
gdp.drop([1,3,7,36,49,61,62,63,64,65,68,73,74,95,98,102,103,104,105,107,110,128,134,135,136,139,140,142,153,156,161,
          170,181,183,191,197,198,204,215,217,218,230,231,236,238,240,241,249,259], axis=0, inplace=True)

#rename columns to match json file
gdp['Country Name'].replace({'Bahamas, The': 'The Bahamas',
                       'Brunei Darussalam': 'Brunei',
                       'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
                       'Congo, Rep.': 'Republic of the Congo',
                       'Czechia': 'Czech Republic',
                       'Egypt, Arab Rep.': 'Egypt',
                       'Gambia, The': 'Gambia',
                       'Guinea-Bissau': 'Guinea Bissau',
                       'Hong Kong SAR, China': 'Hong Kong',
                       'Iran, Islamic Rep.': 'Iran',
                       'Kyrgyz Republic': 'Kyrgyzstan',
                       'Korea, Rep.': 'South Korea',
                       'Lao PDR': 'Laos',
                       'North Macedonia': 'Macedonia',
                       'West Bank and Gaza': 'West Bank',
                       'Russian Federation': 'Russia',
                       'Syrian Arab Republic': 'Syria',
                       'Timor-Leste': 'East Timor',
                       'Turkiye': 'Turkey',
                       'United States': 'United States of America',
                       'Venezuela, RB': 'Venezuela',
                       'Serbia': 'Republic of Serbia',
                       'Tanzania': 'United Republic of Tanzania',
                       'Yemen, Rep.': 'Yemen'}, inplace=True)
#also including Cote d'Ivoire -> Ivory Coast and Korea Dem. People's Rep. -> North Korea




######6
#missing values
gdp2021.isnull().sum()
gdp2021_na = gdp2021[gdp2021['2021'].isna()]

#for missing GDP values, will use the latest year for all countries on list
#I couldn't find a way to replace a custom list in python so I did so in excel
#Channel Islands (2007): 11515260000, Cuba (2020): 107352000000, Eritrea (2011): 2065000000, Gibraltar: np.nan, 
#Greenland (2020): 3076020000, Isle of Man (2019): 7315390000, Not Classified: np.nan, Kuwait (2020): 105960230000,
#Liechtenstein (2020): 6113950000, St Martin (French, 2014): 772950000, N Marian Islands (2019): 1182000000, 
#N Korea: np.nan, San Marino (2020): 1541200000, S Sudan (2015): 11997800000, St Martin (Dutch, 2018): 1185470000, 
#Syria (2020): 11079800000, Turkmenistan (2019): 45231430000, Venezuela (2014): 482359320000, 
#British Virgin Islands: np.nan, U.S Virgin Islands (2020): 4204000000, Yemen (2018): 21606160000
gdp.to_csv(os.path.join(path, 'Data', 'gdp_unfilled.csv'))

gdp_fill = pd.read_csv(os.path.join(path, 'Data', 'gdp_filled.csv'))
gdp_fill2021 = gdp_fill[["Country Name", "Country Code", "Indicator Name", "Indicator Code", "2021"]]


#check number of na values
gdp_fill2021.isnull().sum()

#drop na values
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



#check distribution of gdp 
sns.histplot(gdp_fill2021['2021 millions'], bins=10, kde = True) #shows extreme values






######7
#create data frame with just countries and gdp for plotting
plot_data = gdp_fill2021[['Country Name', '2021 millions']]

#set up folium map
map = folium.Map(location= [100,0], zoom_start=1.5)

folium.Choropleth(geo_data = country_geo, data = plot_data,
                  columns = ['Country Name', '2021 millions'],
                  key_on = 'feature.properties.name', 
                  fill_color = 'RdYlGn', fill_opacity=0.6, line_opacity=0.1, bins = 7,
                  legend_name='GDP (millions USD)').add_to(map)
folium.LayerControl().add_to(map)

map.save('gdpmap.html')





######8
#8a. This map begins to answer part of one of my research questions:
#How do country characteristics, such as gdp, birth rate, death rate, population, etc. alter total emissions 
#and emissions from different sources? I have narrowed down which countries such as the US, China, and Germany
#that have the highest GDPs of all countries on the map, now I just need to compare these GDP values to 
#GHG emissions to see if there is a correlation, which I think there likely will be.

#8b. This analysis does not lead me to any additional research questions since this was a variable of interest
#in a previous exercise that I created. I think population, as I mentioned in the question above, will also
#be an interesting analysis.