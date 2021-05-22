#import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from sklearn import svm
from import_data import import_satellite_data, import_tree_carbon_data

def main():

    # Set the style
    plt.style.use('fivethirtyeight')

if __name__ == "__main__":
    main()
