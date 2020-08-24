# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:01:06 2020

@author: iboye
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from Utils import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

###### Get charactatistics for each age/gender group and plot the normalized score for each topic label

sns.set(style="ticks", color_codes=True)

####Import and prepare gender and age data
data_age = pd.read_csv("Age_data.csv", encoding='latin-1', sep=',', decimal = '.')
data_age = data_age.iloc[1:]
data_age = data_age.rename(columns = {'arbejdsmarkedsanalyse_koen_alder' : 'Group', 'Unnamed: 1' : 'Question Label', 
                                'Unnamed: 12' : 'Score' , 'Unnamed: 2' : 'Topic Label'})
data_age = data_age[['Group', 'Score', 'Question Label', 'Topic Label']]
data_age = data_age[~data_age['Group'].isin(['Mænd', 'Kvinder'])]
data_age['Score'] = data_age['Score'].str.replace(',', '.').astype(float)

#### Load lookup file, which contains 'Question Label', 'Høj score godt' and 'Remove'
lookup = pd.read_csv("Age_gender_use.csv", encoding='latin-1', sep=',', decimal = '.')
lookup = lookup[['Question Label', ' Remove ', 'Høj score godt']].iloc[:-6]

###Join lookup and data_age table, remove all rows where 'Remove' == 1
data_joined = pd.merge(data_age, lookup, on = 'Question Label', how = 'left')
data_joined = data_joined.loc[data_joined[' Remove '] != 1,['Group', 'Score', 'Question Label', 'Høj score godt', 'Topic Label']]

### Create meta dataframe for looking up 'Høj score godt'###
spm_metadata = data_joined.groupby('Question Label')['Høj score godt'].apply(max)

######## Normlize and invert order of data with 'Høj score godt' == 0 ########
### Pivot data
data_normalized = data_joined.drop(columns = ['Høj score godt', 'Topic Label']).pivot(index = 'Group', columns = 'Question Label')
data_normalized.columns = data_normalized.columns.get_level_values(1)

### Fill NaN values with the median for individual questions
data_normalized = data_normalized.apply(lambda x: x.fillna(x.median()), axis = 1)

### Lookup in the meta dataframe (spm_metadata) and apply the function invert_order to all rows where 'Høj score godt' == 0
data_normalized.loc[:,spm_metadata.loc[spm_metadata == 0].index] = data_normalized.loc[:,spm_metadata.loc[spm_metadata == 0].index].apply(invert_order, axis=1)

### Normalize
data_normalized.loc[:, :] = StandardScaler().fit_transform(data_normalized)
data_normalized.set_index(data_normalized.index)

data_normalized['Group'] = data_normalized.index

### Undo pivot
data_normalized = data_normalized.set_index(['Group']).stack().reset_index(name = 'Normalized Score').rename(columns = {'level_1': 'Question Label'})
data_normalized = data_normalized.set_index('Question Label').join(data_age.loc[:,['Question Label', 'Topic Label']].drop_duplicates().set_index('Question Label')).reset_index()

### Take the mean of the normalized data and group by topic label
data_grouped = data_normalized.groupby(['Group', 'Topic Label'])['Normalized Score'].mean().reset_index()
data_grouped = data_grouped.pivot(index = 'Group', columns = 'Topic Label')


### Print the topic labels for the top or bottom three scores per age/gender group
for row in data_grouped.index:
    print(row)
    print(data_grouped.columns.values[np.argsort(data_grouped.loc[row,:]).values[:4]]) ## -3: de tre højeste, :3 de tre laveste

"""
### Plot the normalized score by age/gender group and topic label
for i in data_grouped.columns:
    data_grouped[i].plot.bar()
    plt.ylabel(i)
    plt.show()
    plt.clf()
"""

