import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, ShuffleSplit
from tqdm import tqdm
from sklearn.metrics import make_scorer
import xgboost

n_estimators_space = [50, 100, 200]
lr_space = [0.1, 0.05, 0.005]
max_depth_space = [1, 2, 3, 5, 10]


max_depths = [1,3,6,8,10]
min_child_weight = [2,5,10]
subsample = [0.4,0.6,0.8]
etas = [0.01, 0.02, 0.1]



FEATURES_DICT = {
    'VIF': ['VH_1', 'VV_1', 'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 'CATEGORY'],
    'SOC_FEATURES': [
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI'
    ],
    'AGB_FEATURES': [
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY", 
    ],
    'ALL': [
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY"
    ]
}

def create_BRT_model(n_estimators, lr, max_depth):
    return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            random_state=0,
            loss='ls'
        )

def create_XGB_model(min_child_weight, max_depth, subsample, eta):
    return xgboost.XGBRegressor(min_child_weight=min_child_weight,
        n_estimators=451,
        max_depth=max_depth,
        eta=eta,
        subsample=subsample)



def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"

def grid_search(feature, pred, isLog):

    data_df = load_csv_to_pd(csv_file_path)

    features_list = FEATURES_DICT[feature]

    ground_truth_col = 'AGB' if pred == 'agb' else 'OC'

    X = data_df[features_list].values.astype(np.float32)
    y = np.log(data_df[ground_truth_col].values.astype(np.float32)) if isLog else data_df[ground_truth_col].values.astype(np.float32)

    scoring = {
        'mean_squared_error': make_scorer(mean_squared_error), 
        'mean_absolute_error': make_scorer(mean_squared_error),
        'r2_score': make_scorer(r2_score)
    }
    cv_splits = 5
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)

    results_df = pd.DataFrame(columns=['idx', 'lr', 'n_estimators', 'max_depth', 'fit_time', 'score_time', 'test_root_mean_squared_error', 'test_mean_absolute_error', 'test_r2_score'])

    idx = 0
    if pred == 'soc':
        for lr, n_estimators, max_depth in tqdm(list(itertools.product(lr_space, n_estimators_space, max_depth_space))):
            brt = create_BRT_model(n_estimators,lr,max_depth)

            scores = cross_validate(brt, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            
            result_dict = {}

            result_dict['idx'] = idx
            result_dict['lr'] = lr
            result_dict['n_estimators'] = n_estimators
            result_dict['max_depth'] = max_depth

            result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
            result_dict['score_time'] = sum(scores['score_time']) / cv_splits
            result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
            result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
            result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
            results_df = results_df.append(result_dict, ignore_index=True)
            print('IDX: {:d} | lr {:.4f} | n_estimators {:.4f} | max_depth {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(idx, lr, n_estimators, max_depth, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
            idx += 1

        results_df.to_csv('../../out/joint_result/brt_OC_gridsearch.csv', index=False)
    else:
        for mcw, md, ssp, eta in tqdm(list(itertools.product(min_child_weight, max_depths, subsample, etas))):
            model = create_XGB_model(mcw,md,ssp,eta)

            scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            
            result_dict = {}

            result_dict['min_child_weight'] = mcw
            result_dict['eta'] = eta
            result_dict['subsample'] = ssp
            result_dict['max_depth'] = md

            result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
            result_dict['score_time'] = sum(scores['score_time']) / cv_splits
            result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
            result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
            result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
            results_df = results_df.append(result_dict, ignore_index=True)
            print('IDX: {:d} | lr {:.4f} | n_estimators {:.4f} | max_depth {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(mcw, eta, ssp, md, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
            idx += 1

        results_df.to_csv('../../out/joint_result/xgb_AGB_gridsearch.csv', index=False)


# grid_search('SOC_FEATURES', 'soc', True)
grid_search('ALL', 'agb', False)