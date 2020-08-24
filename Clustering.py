# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:28:27 2020

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

##Based on the average answer of each group for each topic label


sns.set(style="ticks", color_codes=True)


##### Import and prepare job data #####
data_job = pd.read_csv("Job_data.csv", encoding='latin-1', sep=';', decimal = ',')
data_job = data_job[['Group', 'Score', 'Question Label', 'Hoej Score Godt', 'Topic Label']]
data_job = data_job[~data_job['Group'].isin(['Uoplyst', 'Total'])]

### Create meta dataframe for looking up 'Høj score godt'###
spm_metadata = data_job.groupby('Question Label')['Hoej Score Godt'].apply(max)


######## Normlize and invert order of data with 'Høj score godt' == 0 ########
### Pivot data
data_normalized = data_job.drop(columns = ['Hoej Score Godt', 'Topic Label']).pivot(index = 'Group', columns = 'Question Label')
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
data_normalized = data_normalized.set_index('Question Label').join(data_job.loc[:,['Question Label', 'Topic Label']].drop_duplicates().set_index('Question Label')).reset_index()

### Take the mean of the normalized data and group by topic label
data_grouped = data_normalized.groupby(['Group', 'Topic Label'])['Normalized Score'].mean().reset_index()
data_grouped = data_grouped.pivot(index = 'Group', columns = 'Topic Label')

### Apply clustering
kmeans = KMeans(n_clusters = 4, random_state = 42)
data_grouped['Cluster'] = kmeans.fit_predict(data_grouped.values)


### Violin plots of the normalized score by clusters, for each topic label
for i in data_grouped.drop(columns = 'Cluster').columns:
    sns.violinplot(x = 'Cluster', y = i, data = data_grouped)
    plt.show()


#### Used Elbow Method to decide on number of clusters. 
#Was a bit inconclusive, but choose 4 due to practicallity 
"""
sse={}
mean_elbow = data_normalized.copy()
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(mean_elbow)
    mean_elbow["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.show()
"""