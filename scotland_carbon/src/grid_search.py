import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, ShuffleSplit
from tqdm import tqdm
from sklearn.metrics import make_scorer
import xgboost
from utils import load_csv_to_pd, FEATURES_DICT


# Hyperparameter space we search through
brt_n_estimators = [x for x in range(100, 3000, 100)]
brt_lrs = [0.1, 0.05, 0.005]
brt_max_depths = [1, 2, 3, 5, 10]

xgb_max_depths = [1,3,6,8,10]
xgb_min_child_weight = [2,5,10]
xgb_subsamples = [0.4,0.6,0.8]
xgb_etas = [0.01, 0.02, 0.1]

rf_n_estimators_space = [x for x in range(100, 3000, 100)]
rf_max_features = [1/2,1/3,1/6,1,2,3,4,5,6,7,8,9,10]

# Location of data source
csv_file_path = r"Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"

# Methods that create the corresponding machine learning model upon providing parameters 
def create_BRT_model(n_estimators, lr, max_depth):
    return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=0,
            loss='ls'
        )

def create_XGB_model(min_child_weight, max_depth, subsample, eta):
    return xgboost.XGBRegressor(
            min_child_weight=min_child_weight,
            n_estimators=451,
            max_depth=max_depth,
            eta=eta,
            subsample=subsample
        )

def create_RF_model(n_estimators, max_features):
    return RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features
        )

# Perform grid search
def grid_search(feature, pred, ml_model, isLog):

    # Load data to pandas data frame
    data_df = load_csv_to_pd(csv_file_path)

    # Obtain the list of features corresponding to the specified model
    features_list = FEATURES_DICT[feature]

    # Get ground truth data name
    ground_truth_col = 'AGB' if pred == 'agb' else 'OC'

    # Create training data
    X = data_df[features_list].values.astype(np.float32)
    # Create ground truth data
    y = np.log(data_df[ground_truth_col].values.astype(np.float32)) if isLog else data_df[ground_truth_col].values.astype(np.float32)

    # Specify scoring metrics
    scoring = {
        'mean_squared_error': make_scorer(mean_squared_error), 
        'mean_absolute_error': make_scorer(mean_absolute_error),
        'r2_score': make_scorer(r2_score)
    }

    # Cross validation splits
    cv_splits = 5
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2)

    # Cross validation results
    results_df = pd.DataFrame(columns=['idx', 'lr', 'n_estimators', 'max_depth', 'min_child_weight', 'max_feature', 'fit_time', 'score_time', 'test_root_mean_squared_error', 'test_mean_absolute_error', 'test_r2_score'])

    idx = 0
    # Grid search for BRT models
    if ml_model == 'brt':

        # We iterate through every combination of hyperparameters
        for lr, n_estimators, max_depth in tqdm(list(itertools.product(brt_lrs, brt_n_estimators, brt_max_depths))):
            # Create BRT model with the corresponding hyperparameter
            brt = create_BRT_model(n_estimators,lr,max_depth)
            # Get validation scores through cross validation
            scores = cross_validate(brt, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            
            # result_dict to store results for this iteration
            result_dict = {}

            # Store the hyperparamters used in this iteration
            result_dict['idx'] = idx
            result_dict['lr'] = lr
            result_dict['n_estimators'] = n_estimators
            result_dict['max_depth'] = max_depth

            # Store the time used in this iteration
            result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
            result_dict['score_time'] = sum(scores['score_time']) / cv_splits
            # Store the evaluation metrics for this set of hyperparameter
            result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
            result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
            result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
            # Append results for this iteration to all results
            results_df = results_df.append(result_dict, ignore_index=True)
            print('IDX: {:d} | lr {:.4f} | n_estimators {:.4f} | max_depth {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(idx, lr, n_estimators, max_depth, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
            idx += 1

    # Grid search for XGBoost models
    elif ml_model == 'xgb' :
        # We iterate through every combination of hyperparameters
        for mcw, md, ssp, eta in tqdm(list(itertools.product(xgb_min_child_weight, xgb_max_depths, xgb_subsamples, xgb_etas))):
            
            # Create XGB model with the corresponding hyperparameter
            model = create_XGB_model(mcw,md,ssp,eta)
            # Get validation scores through cross validation
            scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            
            # result_dict to store results for this iteration
            result_dict = {}

            # Store the hyperparamters used in this iteration
            result_dict['min_child_weight'] = mcw
            result_dict['eta'] = eta
            result_dict['subsample'] = ssp
            result_dict['max_depth'] = md
            # Store the time used in this iteration
            result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
            result_dict['score_time'] = sum(scores['score_time']) / cv_splits
            # Store the evaluation metrics for this set of hyperparameter
            result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
            result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
            result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
            # Append results for this iteration to all results
            results_df = results_df.append(result_dict, ignore_index=True)
            print('IDX: {:d} | lr {:.4f} | n_estimators {:.4f} | max_depth {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(mcw, eta, ssp, md, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
            idx += 1
    
    # Grid search for RF models
    elif ml_model == 'rf':
        for n_estimators, max_fea in tqdm(list(itertools.product(rf_n_estimators_space, rf_max_features))):
            
            # Create BRT model with the corresponding hyperparameter
            rf = create_RF_model(n_estimators,max_fea)
           
            # Get validation scores through cross validation
            scores = cross_validate(rf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            
            # result_dict to store results for this iteration
            result_dict = {}
            # Store the hyperparamters used in this iteration
            result_dict['idx'] = idx
            result_dict['n_estimators'] = n_estimators
            result_dict['max_feature'] = max_fea
            # Store the time used in this iteration
            result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
            result_dict['score_time'] = sum(scores['score_time']) / cv_splits
            # Store the evaluation metrics for this set of hyperparameter
            result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
            result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
            result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
            # Append results for this iteration to all results
            results_df = results_df.append(result_dict, ignore_index=True)
            print('IDX: {:d} | n_estimators {:.4f} | max_feature {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(idx, n_estimators, max_fea, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
            idx += 1
    
    # Save result to csv
    results_df.to_csv(feature + '_' + pred + '_' + ml_model + '_gridsearch.csv', index=False)

grid_search('MODEL_A', 'soc', 'xgb', True)
grid_search('MODEL_D', 'agb', 'rf', False)