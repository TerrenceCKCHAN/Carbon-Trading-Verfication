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

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats

from rasterio.plot import show

import numpy as np

import pickle

from import_data import import_data, import_satellite_data, import_tree_carbon_data, import_grid, import_small_region, import_medium_region, import_large_region, import_region_grid, import_sentinel_grid, import_landsat_grid
from feature_engineering import log_transform, standardization, dim_reduction
from plots import draw_map, print_tree, plot_var_rank, plot_results, plot_scatter

INVEN = ['nonwood', 'wood', 'as', 'br', 'co', 'fe', 'gr', 'ot', 'yo']
INVEN_TRAIN = ['non', 'wood', 'as', 'ba', 'br', 'co', 'fa', 'fe', 'gra',
            'gro',  'lo', 'mib', 'mic', 'op', 'ot', 'qu', 'ro', 'sh', 'ur', 'wib', 'yo']
PDA_RF = []

def evaluate(model, x_test, y_test):
    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'Mg C/ha.')
    mse = metrics.mean_squared_error(y_test, predictions)
    print('RMSE:', round(math.sqrt(mse), 2), 'Mg C/ha.')

    r2 = metrics.r2_score(y_test, predictions)
    print('R2:', round(math.sqrt(r2), 2))

    y_test[y_test == 0] = 1
    mape = np.mean(100 * (errors / y_test))
    print('Accuracy:', round(model.score(x_test, y_test)*100, 2), '%.')

    return predictions, errors, [mse, r2, mape]

def data_transformation(data, labels, settings):
    if settings[0] == True:
        #Capping the outlier rows with Percentiles
        upper_lim = np.quantile(labels, .95)
        print(f'Upper limit {upper_lim}')
        labels[labels > upper_lim] = upper_lim

    if settings[1] == True:
        # Take log
        data = log_transform(data)

    #data = standardization(data)


    x_train, x_test, y_train, y_test =  train_test_split(data, labels, test_size=0.25, random_state=42)

    # Dimensionality reduction
    x_train, x_test = dim_reduction(x_train, x_test, y_train, settings[3])

    return x_train, x_test, y_train, y_test


def train_model(data, labels, filename, model_name, settings):

    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=2900)
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(min_child_weight=2, n_estimators=451, max_depth=6, eta=0.01, subsample=1, colsample_bytree=0.6)
        if not isinstance(data,(list,pd.core.series.Series,np.ndarray)):
            data = data.to_numpy()
        if not isinstance(x_train,(list,pd.core.series.Series,np.ndarray)):
            x_train = x_train.to_numpy()
        if not isinstance(x_test,(list,pd.core.series.Series,np.ndarray)):
            x_test = x_test.to_numpy()
        #print(isinstance(data,(list,pd.core.series.Series,np.ndarray)))
    else:
        model = LR()

    # hyperparameters:
        # n_estimators: The number of trees in the ensemble, often increased until no further improvements are seen.
        # max_depth: The maximum depth of each tree, often values are between 1 and 10.
        # eta: The learning rate used to weight each model, often set to small values such as 0.3, 0.1, 0.01, or smaller.
        # subsample: The number of samples (rows) used in each tree, set to a value between 0 and 1, often 1.0 to use all samples.
        # colsample_bytree: Number of features (columns) used in each tree, set to a value between 0 and 1, often 1.0 to use all features.

    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # evaluate model
    scores = cross_val_score(model, data, labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    mae = scores.mean()
    std = scores.std()
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )
    # Train the model on training data
    model.fit(x_train, y_train)

    model_save = filename  + '.sav'
    pickle.dump(model, open(model_save, 'wb'))
    #model = pickle.load(open(model_save, 'rb'))
    predictions, errors, scores = evaluate(model, x_test, y_test)
    scores = scores.append([mae, std])

    plot_results(predictions, errors, y_test, filename)
    plot_scatter(predictions, y_test, model_name, filename)

    #draw_map(predictions)

    return model, scores

def run(settings, model_name, filename, region, tag):

    data, labels = import_region_grid(region, tag)
    labels = np.log(labels.values.ravel()+1+labels.values.ravel().min())
    avg_carbon = labels.mean()

    print('Average Carbon:', round(np.mean(avg_carbon), 2), 'Mg C/ha.')

    feature_list = list(data.columns)
    file_dir = 'output/'+ region+'_region/' + model_name +'/' + filename + tag
    model, scores = train_model(data, labels, file_dir, model_name, settings)
    #print_tree(model[2], filename, feature_list)
    var_rank = plot_var_rank(model, model_name, feature_list, file_dir)

    # run_raster(model, model_name, region)
    # draw_map(model_name, region, file_dir)

    return  model, scores, var_rank

def pca_vs_pda(data, labels):
    print('NO CAP, NO BIN')
    run(data, labels, 'noCap_noBin_noCross')
    print('NO CAP, PDA')
    run(data, labels, 'noCap_pda_noCross', 'pda')
    print('NO CAP, PCA')
    run(data, labels, 'noCap_pca_noCross', 'pca')

    #Capping the outlier rows with Percentiles
    upper_lim = np.quantile(labels, .95)
    print(f'Upper limit {upper_lim}')

    labels[labels > upper_lim] = upper_lim
    print('CAP, NO BIN')
    run(data, labels, 'capped_noBin_noCross')
    print('CAP, PDA')
    run(data, labels, 'capped_pda_noCross', 'pda')
    print('CAP, PCA')
    run(data, labels, 'capped_pca_noCross', 'pca')

def run_raster(model, model_name, region):
    output = 'output/' + region + '_region/raster/prediction_map_'+model_name+'.tif'
    input  = 'data/' + region + '_region/raster/combined.tif'

    sentinel_raster = rasterio.open('data/' + region + '_region/raster/' + region + '_sentinel.tif')
    profile = sentinel_raster.profile

    landsat_2 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat2.tif')
    landsat_3 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat3.tif')
    landsat_4 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat4.tif')
    landsat_5 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat5.tif')
    landsat_6 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat6.tif')
    landsat_7 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat7.tif')

    #inven = rasterio.open('data/' + region + '_region/raster/' + region + '_inven.tif')
    inventory = []
    if  region == 'train':
        inven = INVEN_TRAIN
    else:
        inven = INVEN
    for species in inven:
        inventory.append(rasterio.open('data/' + region + '_region/raster/' + region + '_' + species + '.tif'))

    ndvi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # ndvi = (band5 - band4) / (band5 + band4)
    savi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # savi = (band5 - band4) / (band5 + band4 + 0.5)
    evi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))
    arvi = np.empty(landsat_4.shape, dtype = rasterio.float32)
    # arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

    profile.update({'count': 8+len(inventory), 'dtype': rasterio.float32})

    with rasterio.open(input, 'w+', **profile) as inp:
        inp.write(sentinel_raster.read(1),1)
        inp.write(sentinel_raster.read(2),2)
        inp.write(landsat_2.read(1),3)
        inp.write(landsat_3.read(1),4)
        inp.write(landsat_4.read(1),5)
        inp.write(landsat_5.read(1),6)
        inp.write(landsat_6.read(1),7)
        inp.write(landsat_7.read(1),8)
        i = 1
        for species in inventory:
            inp.write(species.read(1), 8+i)
            i += 1
        # inp.write(non.read(1),9)
        # inp.write(wood.read(1),10)
        # inp.write(as.read(1),11)
        # inp.write(br.read(1),12)
        # inp.write(co.read(1),13)
        # inp.write(fe.read(1),14)
        # inp.write(gr.read(1),15)
        # inp.write(ot.read(1),16)
        # inp.write(yo.read(1),17)

        profile.update({'count': 1})
        with rasterio.open(output, 'w', **profile) as dst:

            # perform prediction on each small image patch to minimize required memory
            patch_size = 6
            for i in range((inp.shape[0] // patch_size) + 1):
                for j in range((inp.shape[1] // patch_size) + 1):
                    # define the pixels to read (and write) with rasterio windows reading
                    window = rasterio.windows.Window(
                        j * patch_size,
                        i * patch_size,
                        # don't read past the image bounds
                        min(patch_size, inp.shape[1] - j * patch_size),
                        min(patch_size, inp.shape[0] - i * patch_size))

                    # read the image into the proper format
                    data = inp.read(window=window)
                    band2 = data[3].astype(float)
                    band2 = np.nan_to_num(band2.reshape(band2.shape[0]*band2.shape[1],1))
                    band4 = data[5].astype(float)
                    band4 = np.nan_to_num(band4.reshape(band4.shape[0]*band4.shape[1],1))
                    band5 = data[6].astype(float)
                    band5 = np.nan_to_num(band5.reshape(band5.shape[0]*band5.shape[1],1))
                    img_swp = np.moveaxis(data, 0, 2)
                    img_flat = img_swp.reshape(-1, img_swp.shape[-1])
                    #img_flat = data.transpose(1,2,0).reshape(data.shape[1]*data.shape[2],13)
                    ndvi = np.where((band5 + band4) == 0., 0., (band5 - band4) / (band5 + band4))
                    savi = np.where((band5 + band4 + 0.5) == 0., 0.,(band5 - band4) / (band5 + band4 + 0.5))
                    evi = np.where((band5 + 6 * band4 - 7.5 * band2 + 1) == 0., 0., 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 + 1)))
                    arvi = np.where((band5 + 2*band4 - band2) == 0., 0., (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2))

                    img_w_ind = np.concatenate([img_flat, ndvi, savi, evi, arvi], axis=1)
                    # remove no data values, store the indices for later use
                    m = np.ma.masked_invalid(img_w_ind)
                    to_predict = img_w_ind.reshape(-1, img_w_ind.shape[-1])

                    # skip empty inputs
                    if not len(to_predict):
                        continue
                    # print(f'img_flat.shape {img_flat.shape}')
                    # print(f'data.shape {data.shape}')
                    # predict
                    #print(type(to_predict))
                    #print(isinstance(to_predict,(list,pd.core.series.Series,np.ndarray)))
                    img_preds = model.predict(to_predict)

                    # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                    # makes the assumption that all bands have identical no-data value arrangements
                    output = np.zeros(img_flat.shape[0])
                    output[~m.mask[:, 0]] = img_preds.flatten()

                    # resize to the original image dimensions
                    #output = output.reshape(*data.shape[1:])
                    output = output.reshape(*img_swp.shape[:-1])
                    # create our final mask
                    #mask = (~m.mask[:, 0]).reshape(*data.shape[1:])
                    mask = (~m.mask[:, 0]).reshape(*img_swp.shape[:-1])

                    # write to the final files
                    dst.write(output.astype(rasterio.float32), 1, window=window)
                    dst.write_mask(mask, window=window)

def run_pda_raster(model, model_name, region):
    output = 'output/' + region + '_region/raster/prediction_map_'+model_name+'.tif'
    input  = 'data/' + region + '_region/raster/combined.tif'

    sentinel_raster = rasterio.open('data/' + region + '_region/raster/' + region + '_sentinel.tif')
    profile = sentinel_raster.profile

    landsat_2 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat2.tif')
    landsat_3 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat3.tif')
    landsat_4 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat4.tif')
    landsat_5 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat5.tif')
    landsat_6 = rasterio.open('data/' + region + '_region/raster/' + region + '_landsat6.tif')
    profile.update({'count': 7, 'dtype': rasterio.float32})

    with rasterio.open(input, 'w+', **profile) as inp:
        inp.write(sentinel_raster.read(1),1)
        inp.write(sentinel_raster.read(2),2)
        inp.write(landsat_2.read(1),3)
        inp.write(landsat_3.read(1),4)
        inp.write(landsat_4.read(1),5)
        inp.write(landsat_5.read(1),6)
        inp.write(landsat_6.read(1),7)

        profile.update({'count': 1})
        with rasterio.open(output, 'w', **profile) as dst:

            # perform prediction on each small image patch to minimize required memory
            patch_size = 6
            for i in range((inp.shape[0] // patch_size) + 1):
                for j in range((inp.shape[1] // patch_size) + 1):
                    # define the pixels to read (and write) with rasterio windows reading
                    window = rasterio.windows.Window(
                        j * patch_size,
                        i * patch_size,
                        # don't read past the image bounds
                        min(patch_size, inp.shape[1] - j * patch_size),
                        min(patch_size, inp.shape[0] - i * patch_size))

                    # read the image into the proper format
                    data = inp.read(window=window)
                    img_swp = np.moveaxis(data, 0, 2)
                    img_flat = img_swp.reshape(-1, img_swp.shape[-1])

                    # remove no data values, store the indices for later use
                    m = np.ma.masked_invalid(img_flat)
                    to_predict = img_flat.reshape(-1, img_flat.shape[-1])

                    # skip empty inputs
                    if not len(to_predict):
                        continue
                    # print(f'img_flat.shape {img_flat.shape}')
                    # print(f'data.shape {data.shape}')
                    # predict
                    #print(type(to_predict))
                    #print(isinstance(to_predict,(list,pd.core.series.Series,np.ndarray)))
                    img_preds = model.predict(to_predict)

                    # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                    # makes the assumption that all bands have identical no-data value arrangements
                    output = np.zeros(img_flat.shape[0])
                    output[~m.mask[:, 0]] = img_preds.flatten()

                    # resize to the original image dimensions
                    #output = output.reshape(*data.shape[1:])
                    output = output.reshape(*img_swp.shape[:-1])
                    # create our final mask
                    #mask = (~m.mask[:, 0]).reshape(*data.shape[1:])
                    mask = (~m.mask[:, 0]).reshape(*img_swp.shape[:-1])

                    # write to the final files
                    dst.write(output.astype(rasterio.float32), 1, window=window)
                    dst.write_mask(mask, window=window)

def feature_search():
    all_results = []
    # outliers, log, normalisation, dim reduction, interp/one hot, filename
    features = [[True, False, '', 'cap'], [False, True, '', 'log'], [False, False, 'pda', 'pda'], [False, False, 'pca','pca'],
                [False, True, '', 'norm'], [False, True, 'pca', 'norm_pca'], [False, True, 'pda', 'norm_pda']]
    region = 'quick'
    models = ['random_forest', 'xgboost']
    tag = '300'
    for setting in features:
        for model_name in models:
            filename = setting[-1]

            print('Setting ' + filename + ' for region ' + region + ' for model ' + model_name + ' with tag ' + tag)

            model, scores, var_rank = run(setting, model_name, filename, region, tag)

            results = [filename, region+tag, model_name]
            results = results.append(scores)
            all_results.append(results)

    results_df = pd.DataFrame(all_results, columns =['Feature', 'Region', 'Model', 'MSE', 'R2', 'MAPE', 'MAE', 'std'])
    data.to_csv('output/results.csv')

    satellite_data = import_satellite_raster()

    actual = import_carbon_raster()
    predictions = model.fit(satellite_data)

def xgb_maxdepth_childweight(params, dtrain, dtest):

    params['eval_metric'] = "mae"
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(2,12,2)
        for min_child_weight in range(1,6)
    ]

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                 max_depth,
                                 min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    return best_params[0], best_params[1], min_mae
    # printed: Best params: 6, 2, MAE: 98.3690782

def eta_sub_tuning(params, dtrain, dtest):

    gridsearch_params = [
        (subsample, eta)
        for subsample in [i/10. for i in range(6,11)]
        for eta in [0.01, 0.05, 0.1, 0.2, 0.3]
    ]
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for subsample, eta in gridsearch_params:
        print("CV with subsample={}, eta={}".format(
                                 subsample,
                                 eta))
        # Update our parameters
        params['subsample'] = subsample
        params['eta'] = eta
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,eta)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    return best_params[0], best_params[1], min_mae
    #Best params: 1.0, 0.6, MAE: 98.3690782

def gamma_tuning(params, dtrain, dtest):

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for gamma in [i/10. for i in range(0,5)]:
        print("CV with gamma={}".format(gamma))
        # Update our parameters
        params['gamma'] = gamma
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = gamma
    print("Best params: {}, MAE: {}".format(best_params, min_mae))
    return best_params, min_mae
    #Best params: 1.0, 0.6, MAE: 98.3690782

def hyperparameter_tuning():
    region = 'quick'
    tag = '300'
    settings = [False, False, '', 'onehot', 'hypertuning']

    data, labels = import_region_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {
        # Parameters that we are going to tune.
        'max_depth':6,
        'min_child_weight': 2,
        'eta':.01,
        'subsample': 1,
        'colsample_bytree': 0.6,
        'gamma':0.2,
        # Other parameters
        'objective':'reg:squarederror',
    }
    eta_vals = [0.01, 0.05, 0.1, 0.2, 0.3]
    params['eval_metric'] = "mae"
    gridsearch_params = [
        (max_depth, min_child_weight, subsample, colsample, gamma, eta)
        for max_depth in range(2,12,2)
        for min_child_weight in range(1,6)
        for subsample in [i/10. for i in range(6,11)]
        for colsample in [i/10. for i in range(6,11)]
        for gamma in [i/10. for i in range(0,5)]
        for eta in eta_vals
    ]

    scores = ['mse', 'r2']
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for max_depth, min_child_weight, subsample, colsample, gamma, eta in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}, subsample={}, colsample={}, gamma={}, eta={}".format(
                                 max_depth,
                                 min_child_weight,
                                 subsample,
                                 colsample,
                                 gamma,
                                 eta))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['subsample'] = subsample
        params['colsample'] = colsample
        params['gamma'] = gamma
        params['eta'] = eta

        start = time.process_time()
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} time taken {}\n".format(mean_mae, boost_rounds, time.process_time() - start))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = params
    print("Best params: max_depth={}, min_child_weight={}, subsample={}, colsample={}, gamma={}, eta={}, MAE: {}".format(best_params['max_depth'], best_params['min_child_weight'], best_params['subsample'], best_params['colsample'], best_params['gamma'], best_params['eta'], min_mae))

    #Best params: 0.01  (eta), MAE: 98.04678799999999
    #Best params: 0 (gamma), MAE: 98.3484922

def hyperparameter_tuning_rf(data, labels):

    settings = [False, False, '', 'onehot', 'hypertuning']

    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(x_train, y_train)
    base_accuracy = evaluate(base_model, x_test, y_test)

    n_estimators=[int(i) for i in range(100, 3100, 100)]
    max_features = [1/6, 1/3, 1/2]
    param_grid = {
        # Parameters that we are going to tune.
        'n_estimators':n_estimators,
        'max_features': max_features,
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model

    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 2)

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, x_test, y_test)
    print(f'Best grid {best_grid}')

    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    #Best params: 0.01  (eta), MAE: 98.04678799999999
    #Best params: 0 (gamma), MAE: 98.3484922

def tune_xgboost(x_train, x_test, y_train, y_test):

    params = {
        # Parameters that we are going to tune.
        'max_depth':10,
        'min_child_weight': 3,
        'gamma': 0.2,
        'eta':.05,
        'subsample': 0.8,
        # Other parameters
        'objective':'reg:squarederror',
    }

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    start = time.process_time()
    max_depth,min_child_weight, mae = xgb_maxdepth_childweight(params, dtrain, dtest)
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    subsample, eta, mae = eta_sub_tuning(params, dtrain, dtest)
    params['subsample'] = subsample
    params['eta'] = eta
    gamma, mae = gamma_tuning(params, dtrain, dtest)
    params['gamma'] = gamma
    print('Taken {} time'.format(time.process_time()-start))
    return params, mae

def tune_all_xgboost():
    region = 'quick'
    tag = '300'
    settings = [False, False, '', 'onehot', 'hypertuning']

    # data, labels = import_region_grid(region, tag)
    # labels = labels.values.ravel()
    # x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    #
    # params = {
    #     # Parameters that we are going to tune.
    #     'max_depth':8,
    #     'min_child_weight': 4,
    #     'gamma': 0.2, #gamma:0
    #     'eta':.05, #eta: 0.01
    #     'subsample': 0.8, #subsample: 0.6
    #     # Other parameters
    #     'objective':'reg:squarederror',
    # }
    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test, label=y_test)
    #
    # subsample, eta, mae = eta_sub_tuning(params, dtrain, dtest)
    # params['subsample'] = subsample
    # params['eta'] = eta
    # gamma, mae = gamma_tuning(params, dtrain, dtest)
    # params['gamma'] = gamma
    # print("Best params COMBINATION: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
    #     params['max_depth'], params['min_child_weight'], params['subsample'], gamma, params['eta'], mae))


    # data, labels = import_sentinel_grid(region, tag)
    # labels = labels.values.ravel()
    # x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    # params, mae = tune_xgboost(x_train, x_test, y_train, y_test)
    # print("Best params SENTINEL 1A: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
    #     params['max_depth'], params['min_child_weight'], params['subsample'], params['gamma'], params['eta'], mae))

    data, labels = import_landsat_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    params, mae = tune_xgboost(x_train, x_test, y_train, y_test)
    print("Best params LANDSAT8: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
        params['max_depth'], params['min_child_weight'], params['subsample'], params['gamma'], params['eta'], mae))


def main():

    region = 'quick'
    tag = '300'


    print(f'Running all data')
    settings = [True, True, 'pca', 'allFeatEng_allData']
    data, labels = import_region_grid(region, tag)
    labels = labels.values.ravel()
    xgb, xgb_scores, xgb_var_rank = run(settings, model_name, filename, region, tag)

    print(f'Running Landsat')
    settings = [True, True, 'pca', 'allFeatEng_landsat']
    data, labels = import_landsat_grid(region, tag)
    labels = labels.values.ravel()
    xgb_landsat, xgb_scores_landsat, xgb_var_rank_landsat = run(settings, model_name, filename, region, tag)

    print(f'Running Sentinel')
    settings = [True, True, 'pca', 'allFeatEng_sentinel']
    data, labels = import_sentinel_grid(region, tag)
    labels = labels.values.ravel()
    xgb_sentinel, xgb_scores_sentinel, xgb_var_rank_sentinel = run(settings, model_name, filename, region, tag)

    print(f'Results for ALL DATA')
    print(f'XGB SCORES {xgb_scores}')

    print(f'Results for LANDSAT')
    print(f'XGB LANDSAT SCORES {xgb_scores_landsat}')

    print(f'Results for SENTINEL')
    print(f'XGB SENTINEL SCORES {xgb_scores_sentinel}')    # model_name = 'xgboost'
    # region = 'quick'
    # tag = '300'
    # settings = [True, False, '', 'onehot', 'hypertuning']
    # filename = 'loglabel'
    # setting = [False, False, '', 'nothing']
    # model, scores, var_rank = run(setting, model_name, filename, region, tag)

    # #model, scores, var_rank = run(setting, model_name, filename, region, tag)
    # setting = [False, False, 'pda', 'nothing']
    # model, scores, var_rank = run(setting, model_name, filename, region, tag)
    # setting = [False, False, 'pca', 'nothing']
    # model, scores, var_rank = run(setting, model_name, filename, region, tag)


    # # model, scores, var_rank = run(setting, model_name, filename, region, '')
    # model, scores, var_rank = run(setting, 'xgboost', filename, region, tag)
    # model, scores, var_rank = run(setting, 'xgboost', filename, region, '')
    # setting = [False, False, 'pca', 'pca']
    # model, scores, var_rank = run(setting, model_name, 'pca', region, tag)
    # model, scores, var_rank = run(setting, model_name, 'pca', region, '')
    # model, scores, var_rank = run(setting, 'xgboost', 'pca', region, tag)
    # model, scores, var_rank = run(setting, 'xgboost', 'pca', region, '')

    # region = 'quick'
    # tag = '300'
    # settings = [False, False, '', 'onehot', 'hypertuning']
    #
    # data, labels = import_region_grid(region, tag)
    # labels = labels.values.ravel()
    # x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    #
    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test, label=y_test)
    #
    # params = {
    #     # Parameters that we are going to tune.
    #     'max_depth':6,
    #     'min_child_weight': 2,
    #     'eta':.01,
    #     'subsample': 1,
    #     'colsample_bytree': 0.6,
    #     # Other parameters
    #     'objective':'reg:squarederror',
    # }
    #
    # model = xgb.train(
    #     params,
    #     dtrain,
    #     num_boost_round=451,
    #     evals=[(dtest, "Test")],
    #     early_stopping_rounds=10
    # )
    #
    # print("Best MAE: {:.2f} with {} rounds".format(
    #              model.best_score,
    #              model.best_iteration+1))
    # # Best MAE: 156.54 with 451 rounds



if __name__ == "__main__":
    main()
