# Boyang Bai
# Date: March 31, 2017
# Description: K-means

import os
import numpy as np
from  sklearn.cluster import KMeans
from matplotlib import pyplot
import pandas as pd

class Kmeans():

    def __init__(self):
        pass

max_iter  = 300
n_clusters = 2
random_state = 170
tol = 0.0001

def loadData(self, file, sheet, cols, skip):
    """Load an excel file to x"""
    x = pd.read_excel(io=file, sheetname=sheet, header=0, parse_cols=cols, skiprows=skip).values
    return x

def K_means(x):
    clf = KMeans(max_iter=max_iter, n_clusters=n_clusters, random_state=random_state, tol=tol)
    clf.fit(x)
    centers = clf.cluster_centers_
    labels = clf.labels_
    return labels

if __name__ == '__main__':
    print("Direct access to " + os.path.basename(__file__))
else:
    print(os.path.basename(__file__) + " class instance")