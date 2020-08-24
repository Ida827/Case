# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:03:56 2020

@author: iboye
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from functions1 import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

def invert_order(x):
    return x*(-1)+max(x)


def print_scores(job_data, age_data, title, age_gender):
    return print('The job score for ' + title + ' is:',  
                 job_data.loc[title, 'Average'].round(decimals = 1),
                 '\n\nThe age/gender score for ' + age_gender + ' is:',
                 age_data.loc[age_gender, 'Average'].round(decimals = 1),
                 '\n\nThe total score is', 
                 (job_data.loc[title, 'Average']+age_data.loc[age_gender, 'Average']).round(decimals = 1),
                 '\n\nThe total scores ranges from ', 
                 (job_data.loc[job_data.Average.idxmin(axis = 0), 'Average'] + age_data.loc[age_data.Average.idxmin(axis = 0), 'Average']).round(decimals = 1),
                 ' to ', (job_data.loc[job_data.Average.idxmax(axis = 0), 'Average'] + age_data.loc[age_data.Average.idxmax(axis = 0), 'Average']).round(decimals = 1))

