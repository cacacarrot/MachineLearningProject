# Boyang Bai
# Date: March 31, 2017
# Description: K-means
# Update: April 4, 2017
# coding=utf8

import pandas as pd
import os
import numpy as np
from  sklearn.cluster import KMeans
from matplotlib import pyplot


# Load original data
filename = 'C:/Users/asus1/Desktop/ML_HW/new_project_data.xlsx'
df = pd.read_excel(filename)
# Create a set of dummy variables from the nominal variables
df_workclass = pd.get_dummies(df['workclass'])
df_marital = pd.get_dummies(df['marital-status'])
df_occupation = pd.get_dummies(df['occupation'])
df_relationship = pd.get_dummies(df['relationship'])
df_race  = pd.get_dummies(df['race'])
df_sex = pd.get_dummies(df['sex'])
df_native = pd.get_dummies(df['native-country'])
# Join the dummy variables to the main dataframe
data = pd.concat([df['age'], df_workclass, df['education-num'],
                    df_marital, df_occupation, df_relationship,
                    df_race, df_sex, df['hours-per-week'],df_native], axis=1)
# K_means
# set parameters
max_iter = 300
n_clusters = 2
random_state = 170
tol = 0.0001
# convert dataframe to array
data_array  = np.array(data)
# fit
clf = KMeans(max_iter=max_iter, n_clusters=n_clusters,
             random_state=random_state, tol=tol)
clf.fit(data_array)
centers = clf.cluster_centers_
labels = clf.labels_
# convert array to dataframe
labels_df = pd.DataFrame(labels)
labels_df.columns = ['predict']
# Create a set of dummy variables from the column "income"
df_income = pd.get_dummies(df['income'])
# join the new columns to df
df_new = df.join(df_income)
df_new = df_new.join(labels_df)
'''
   df_new.to_csv('K_means.csv')
'''
df_group = df_new.groupby([' <=50K', 'predict'])
print df_group.age.count()