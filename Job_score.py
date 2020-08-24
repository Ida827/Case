# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:20:25 2020

@author: iboye
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from Utils import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings


###### Bin the answers and make an average score for each job and age/gender group

sns.set(style="ticks", color_codes=True)


########################### Job score ########################


##### Import and prepare job data #####
data_job = pd.read_csv("Job_data.csv", encoding='latin-1', sep=';', decimal = ',')
data_job = data_job[['Group', 'Score', 'Question Label', 'Hoej Score Godt', 'Topic Label']]
data_job = data_job[~data_job['Group'].isin(['Uoplyst', 'Total'])]


### Create meta dataframe for looking up 'Høj score godt'###
spm_metadata = data_job.groupby('Question Label')['Hoej Score Godt'].apply(max)

######## Normlize and invert order of data with 'Høj score godt' == 0 ########
### Pivot data
data_job = data_job.drop(columns = ['Hoej Score Godt', 'Topic Label']).pivot(index = 'Group', columns = 'Question Label')
data_job.columns = data_job.columns.get_level_values(1)

### Fill NaN values with the median for individual questions
data_job = data_job.apply(lambda x: x.fillna(x.median()), axis = 1)

### Lookup in the meta dataframe (spm_metadata) and apply the function invert_order to all rows where 'Høj score godt' == 0
data_job.loc[:,spm_metadata.loc[spm_metadata == 0].index] = data_job.loc[:,spm_metadata.loc[spm_metadata == 0].index].apply(invert_order, axis=1)

### Normalize
data_job.loc[:, :] = StandardScaler().fit_transform(data_job)
data_job.set_index(data_job.index)

### Bin the data and create an Average column
data_job = data_job.apply(lambda x: pd.cut(x, bins = 10, labels = False, precision = 7), axis = 1)
data_job['Average'] = data_job.mean(axis = 1)
data_job.loc[:,:] = MinMaxScaler(feature_range = (0, 5)).fit_transform(data_job)

## Plot a histogram and save as .svg file
#data_job.Average.hist(bins = 10)
#plt.savefig('Histogram_job_score.svg', format = 'svg')



####################### Age and gender score #######################

####Import and prepare gender and age data
data_age = pd.read_csv("Age_data.csv", encoding='latin-1', sep=',', decimal = '.')
data_age = data_age.iloc[1:]
data_age = data_age.rename(columns = {'arbejdsmarkedsanalyse_koen_alder' : 'Group', 'Unnamed: 1' : 'Question Label', 
                                'Unnamed: 12' : 'Score' })
data_age = data_age[['Group', 'Score', 'Question Label']]
data_age = data_age[~data_age['Group'].isin(['Mænd', 'Kvinder'])]
data_age['Score'] = data_age['Score'].str.replace(',', '.').astype(float)

#### Load lookup file, which contains 'Question Label', 'Høj score godt' and 'Remove'
lookup = pd.read_csv("Age_gender_use.csv", encoding='latin-1', sep=',', decimal = '.')
lookup = lookup[['Question Label', ' Remove ', 'Høj score godt']].iloc[:-6]

###Join lookup and data_age table, remove all rows where 'Remove' == 1
data_joined = pd.merge(data_age, lookup, on = 'Question Label', how = 'left')
data_joined = data_joined.loc[data_joined[' Remove '] != 1,['Group', 'Score', 'Question Label', 'Høj score godt']]

### Create meta dataframe for looking up 'Høj score godt'###
spm_metadata = data_joined.groupby('Question Label')['Høj score godt'].apply(max)

######## Normlize and invert order of data with 'Høj score godt' == 0 ########
### Pivot data
data_joined = data_joined.drop(columns = ['Høj score godt']).pivot(index = 'Group', columns = 'Question Label')
data_joined.columns = data_joined.columns.get_level_values(1)

### Fill NaN values with the median for individual questions
data_joined = data_joined.apply(lambda x: x.fillna(x.median()), axis = 1)

### Lookup in the meta dataframe (spm_metadata) and apply the function invert_order to all rows where 'Høj score godt' == 0
data_joined.loc[:,spm_metadata.loc[spm_metadata == 0].index] = data_joined.loc[:,spm_metadata.loc[spm_metadata == 0].index].apply(invert_order, axis=1)

### Normalize
data_joined.loc[:, :] = StandardScaler().fit_transform(data_joined)
data_joined.set_index(data_joined.index)

### Bin the data and create an Average column
data_joined = data_joined.apply(lambda x: pd.cut(x, bins = 10, labels = False, precision = 7), axis = 1)
data_joined['Average'] = data_joined.mean(axis = 1)
data_joined.loc[:,:] = MinMaxScaler(feature_range = (0, 5)).fit_transform(data_joined)

## Plot a histogram and save as .svg file
#data_joined.Average.hist(bins = 10)
#plt.savefig('Histogram_age_score.svg', format = 'svg')

###### Print the job score, age score and total score
print(data_joined['Average'])
print_scores(data_job, data_joined, 'Specialpædagoger', 'Kvinder, 18 - 24 år')




