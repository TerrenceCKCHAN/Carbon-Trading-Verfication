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

from import_data import import_satellite_data, import_tree_carbon_data

# ------------------- Scaling data ----------------------------------------

def log_transform(data):
    # select the float columns
    float_features = data.select_dtypes(include=[np.float]).columns
    for feature in float_features:
        # print(f'{feature} before {data.loc[:,feature]}')
        data.loc[:,feature] = np.log(data.loc[:,feature].ravel()+1+abs(data.loc[:,feature].min()))
        # print(f'{feature} after {data.loc[:,feature]}')
    return data

def standardization(data):
    float_features = data.select_dtypes(include=[np.float]).columns
    print(float_features)
    for feature in float_features:
        # print(f'{feature} before {data.loc[:,feature]}')
        data.loc[:,feature] = (data.loc[:,feature] - data.loc[:,feature].mean()) / data.loc[:,feature].std()
        # print(f'{feature} after {data.loc[:,feature]}')

    return data
# ------------- Dimensionality reduction ---------------------------------

def dim_reduction(train_features, test_features, train_labels, type):
    if type == 'pca':
        print('PCA')
        pca = PCA(n_components = 7)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
    elif type == 'pda':
        print('PDA')
        pda = PDA(n_components = 7)
        train_features = pda.fit_transform(train_features, train_labels)
        test_features = pda.transform(test_features)
    return train_features, test_features

def main():
    data, labels = import_satellite_data()

    data = log_transform(data)
if __name__ == "__main__":
    main()
