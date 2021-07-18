#import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as PDA


from scipy.stats.mstats import winsorize

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from os import path as op
import pickle

import geopandas as gpd
import shapely as shp

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats

from rasterio.plot import show

import numpy as np


# ------------------- Scaling data ----------------------------------------

def log_transform(data):
    # Select the float columns
    float_features = data.select_dtypes(include=[np.float]).columns
    for feature in float_features:
        data.loc[:,feature] = np.log(data.loc[:,feature].ravel()+1+abs(data.loc[:,feature].min()))
    return data

# ------------- Dimensionality reduction ---------------------------------

def dim_reduction(train_features, test_features, train_labels, settings, n_components):
    # If there is no inven and the dataset is sentinel, there are only two features
    if settings['dataset']  ==  'sentinel' and settings['inven'] == False:
        n_components = 2
    if settings['dim'] == 'pca':
        print('PCA')
        pca = PCA(n_components = n_components)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
    elif settings['dim'] == 'pda':
        print('PDA')
        pda = PDA(n_components = n_components)
        train_features = pda.fit_transform(train_features, train_labels)
        test_features = pda.transform(test_features)
    return train_features, test_features

def main():
    print("Feature Engineering main")
if __name__ == "__main__":
    main()
