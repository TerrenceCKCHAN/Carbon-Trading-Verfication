#import sklearn
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as PDA
from sklearn import metrics
from sklearn.linear_model import LinearRegression as LR

import xgboost as xgb

import time
import math
import os

from scipy.stats.mstats import winsorize

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from os import path as op
import pickle

import pandas as pd
import geopandas as gpd
import shapely as shp
import tempfile

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
from rasterio.io import MemoryFile
import rasterstats

from rasterio.plot import show

import numpy as np

import pickle

from import_data import import_grid
from feature_engineering import log_transform, dim_reduction
from plots import draw_map, print_tree, plot_results, plot_scatter, var_importance, histogram, plot_topVar
from model_params import get_params, get_inven_abv

def evaluate(model, x_test, y_test):
    # Predict on the test data
    predictions = model.predict(x_test)

    # Calculate the absolute errors
    errors = abs(predictions - y_test)

    mse = metrics.mean_squared_error(y_test, predictions)
    print('RMSE:', round(math.sqrt(mse), 2), 'Mg C/ha.')

    r2 = metrics.r2_score(y_test, predictions)
    print('R2:', round(math.sqrt(r2), 2))

    nonzero_y_test = np.where(y_test == 0, 1, y_test)
    mape = np.mean(100 * (errors / nonzero_y_test))
    print('Accuracy:', round(model.score(x_test, nonzero_y_test)*100, 2), '%.')

    return predictions

def data_transformation(data, labels, settings):
    if settings['cap'] != 0:
        cap = 1 - settings['cap']
        #Capping the outlier rows with Percentiles
        upper_lim = np.quantile(labels, cap)
        print(f'Upper limit {upper_lim}')
        labels[labels > upper_lim] = upper_lim

    if settings['log'] == True:
        # Take log
        data = log_transform(data)

    #data = standardization(data)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    # Dimensionality reduction
    # Set number of used features
    n_components = 7
    x_train, x_test = dim_reduction(x_train, x_test, y_train, settings, n_components)

    return x_train, x_test, y_train, y_test

def get_model(settings):
    params = get_params(settings)
    model_name = settings['model']
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                    max_features=params['max_features'])
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(min_child_weight=params['min_child_weight'],
                                n_estimators=451,
                                max_depth=params['max_depth'],
                                eta=params['eta'],
                                subsample=params['subsample'])
    else:
        model = LR()

    return model

def train_model(data, labels, settings):
    data = np.nan_to_num(data)
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    model = get_model(settings)

    # Convert data to numpy
    if not isinstance(data,(list,pd.core.series.Series,np.ndarray)):
        data = data.to_numpy()
    if not isinstance(x_train,(list,pd.core.series.Series,np.ndarray)):
        x_train = x_train.to_numpy()
    if not isinstance(x_test,(list,pd.core.series.Series,np.ndarray)):
        x_test = x_test.to_numpy()

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # Evaluate model
    scores = cross_val_score(model, data, labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # Force scores to be positive
    scores = np.absolute(scores)
    mae = scores.mean()
    std = scores.std()
    data_sum = np.sum(scores)
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

    # Train the model on training data
    model.fit(x_train, y_train)

    # Save model
    filename = settings['filename']
    model_save = filename + '.sav'
    pickle.dump(model, open(model_save, 'wb'))

    # Load model
    #model = pickle.load(open(model_save, 'rb'))

    predictions = evaluate(model, x_test, y_test)

    # Plots predictions, labels and errors --> currently too many data points to visualise clearly
    # plot_results(predictions, errors, y_test, filename)

    # Scatter plots
    plot_scatter(predictions, y_test, settings)

    return model

def run(settings):

    settings['filename'] = settings['filename']+'_'+settings['dataset']
    data, labels = import_grid(settings)
    # Convert labels to numpy array
    labels = labels.values.ravel()

    # Use log carbon
    if settings['log_label']:
        labels = np.log(labels+1+labels.min())

    avg_carbon = labels.mean()
    print('Average Carbon:', round(np.mean(avg_carbon), 2), 'Mg C/ha.')
    print("CheckPoint 1")
    file_dir = 'output/'+ settings['region']+'_region/' + settings['model'] +'/' + settings['filename']
    print("CheckPoint 2")
    settings['filename'] = file_dir
    print("Entering Model")
    print("Output_file directory: ", file_dir)
    model = train_model(data, labels, settings)
    print("Leaving Model")
    feature_list = list(data.columns)
    # Plot the top features used by model
    var_importance(model, settings, feature_list)
    top_var = 15
    if settings['model'] != 'linear':
        plot_topVar(model, settings, top_var)

    if settings['dim'] == '':
        # Calculate raster
        prediction_raster(model, settings)
        # Produce Carbon Map
        draw_map(settings)

def prediction_raster(model, settings):

    model_name = settings['model']
    filename = settings['filename']
    region = settings['region']
    dataset = settings['dataset']

    input  = 'data/raster/combined.tif'
    output = filename+'.tif'

    # Load satellite raster files
    sentinel_raster = rasterio.open('data/' + region + '_region/raster/' + region + '_sentinel.tif')
    profile = sentinel_raster.profile

    landsat_2 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat2.tif')
    landsat_3 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat3.tif')
    landsat_4 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat4.tif')
    landsat_5 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat5.tif')
    landsat_6 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat6.tif')
    landsat_7 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat7.tif')

    # Load inventory raster files
    inventory = []
    inven = get_inven_abv(settings)
    for species in inven:
        inventory.append(rasterio.open('data/' + region + '_region/raster/' + region + '_' + species + '.tif'))

    # Create empty rasters for vegetation indices
    ndvi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # ndvi = (band5 - band4) / (band5 + band4)
    savi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # savi = (band5 - band4) / (band5 + band4 + 0.5)
    evi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))
    arvi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

    if dataset == 'combined':
        bands = 8+len(inventory)
    elif dataset == 'landsat':
        bands = 6+len(inventory)
    else:
        bands = 2+len(inventory)

    # Update raster file with number of bands used (depends on data input)
    profile.update({'count': bands, 'dtype': rasterio.float32})

    # Keep track of the band placement
    i = 1
    # Open model input file to merge feature rasters
    with rasterio.open(input, 'w+', **profile) as inp:
        if dataset != 'landsat':
            inp.write(sentinel_raster.read(1),i)
            inp.write(sentinel_raster.read(2),i + 1)
            i += 2
        if dataset != 'sentinel':
            inp.write(landsat_2.read(1), i)
            inp.write(landsat_3.read(1), i+1)
            inp.write(landsat_4.read(1), i+2)
            inp.write(landsat_5.read(1), i+3)
            inp.write(landsat_6.read(1), i+4)
            inp.write(landsat_7.read(1), i+5)
            i += 6

        for species in inventory:
            inp.write(species.read(1), i)
            i += 1

        profile.update({'count': 1})
        with rasterio.open(output, 'w', **profile) as dst:

            # Perform prediction on each small image patch to minimize required memory
            window_size = 6
            for i in range((inp.shape[0] // window_size) + 1):
                for j in range((inp.shape[1] // window_size) + 1):
                    # define the pixels to read (and write) with rasterio windows reading
                    window = rasterio.windows.Window(
                        j * window_size,
                        i * window_size,
                        # don't read past the image bounds
                        min(window_size, inp.shape[1] - j * window_size),
                        min(window_size, inp.shape[0] - i * window_size))

                    # read the image into the proper format
                    data = inp.read(window=window)
                    img_shift = np.moveaxis(data, 0, 2)
                    img_flatten = img_shift.reshape(-1, img_shift.shape[-1])
                    img_vegs = img_flatten
                    # Add vegetation indices if there are Landsat bands
                    if dataset != 'sentinel':
                        band2 = data[3].astype(float)
                        band2 = np.nan_to_num(band2.reshape(band2.shape[0]*band2.shape[1],1))
                        band4 = data[5].astype(float)
                        band4 = np.nan_to_num(band4.reshape(band4.shape[0]*band4.shape[1],1))
                        band5 = data[6].astype(float)
                        band5 = np.nan_to_num(band5.reshape(band5.shape[0]*band5.shape[1],1))

                        ndvi = np.where((band5 + band4) == 0., 0., (band5 - band4) / (band5 + band4))
                        savi = np.where((band5 + band4 + 0.5) == 0., 0.,(band5 - band4) / (band5 + band4 + 0.5))
                        evi = np.where((band5 + 6 * band4 - 7.5 * band2 + 1) == 0., 0., 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 + 1)))
                        arvi = np.where((band5 + 2*band4 - band2) == 0., 0., (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2))
                        img_vegs = np.concatenate([img_flatten, ndvi, savi, evi, arvi], axis=1)

                    # Remove no data values, store the indices for later use
                    m = np.ma.masked_invalid(img_vegs)
                    pred_input = img_vegs.reshape(-1, img_vegs.shape[-1])

                    # Skip empty inputs
                    if not len(pred_input):
                        continue

                    pred_out = model.predict(pred_input)

                    # Revert predictions if log labels were used
                    if settings['log_label']:
                        pred_out = np.exp(pred_out)

                    # Add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                    # Makes the assumption that all bands have identical no-data value arrangements
                    output = np.zeros(img_flatten.shape[0])
                    output[~m.mask[:, 0]] = pred_out.flatten()

                    # Resize to the original image dimensions
                    output = output.reshape(*img_shift.shape[:-1])

                    # Create the final mask
                    mask = (~m.mask[:, 0]).reshape(*img_shift.shape[:-1])

                    # Write to the final files
                    dst.write(output.astype(rasterio.float32), 1, window=window)
                    dst.write_mask(mask, window=window)
        inp.close()

def main():

    # settings = {
    #     'dataset': 'combined',
    #     'region': 'quick',
    #     'model': 'random_forest',
    #     'inven': True,
    #     'filename': 'tuned_noFEng',
    #     'cap': 0,
    #     'log': False,
    #     'dim': '',
    #     'log_label': False,
    # }

    settings = {
        'dataset': 'combined',
        'region': 'quick',
        'model': 'linear',
        'inven': True,
        'filename': 'tuned_noFEng',
        'cap': 0,
        'log': False,
        'dim': '',
        'log_label': False,
    }
    run(settings)


if __name__ == "__main__":
    main()
