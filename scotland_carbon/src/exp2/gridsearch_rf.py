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

# n_estimators_space = [50, 100, 200]
# lr_space = [0.1, 0.05, 0.005]
# max_depth_space = [1, 2, 3, 5, 10]


# max_depths = [2,4,6,8,10]
# min_child_weight = [1,2,3,4,5]
# subsample = [0.6,0.7,0.8,0.9,1.0]
# etas = [0.01, 0.05, 0.1, 0.2, 0.3]

# rf_n_estimators_space = [x for x in range(100, 3000, 100)]
# rf_max_features = [1/6, 1/3, 1/2]


rf_n_estimators_space = [300]
rf_max_features = [7]

# 5 7 9 10 
# 100-1000


FEATURES_DICT = {
    'ALL+SOC+22': [
        'VH_1','VV_1',
        'BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_5", "L_6","L_7","L_10","L_11", "CATEGORY",
        "OC", "SG_15_30"
    ],
    'SENT+DEM+AGB': [
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        'AGB'
    ],
}


def create_RF_model(n_estimators, max_features):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features)




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
        'mean_absolute_error': make_scorer(mean_absolute_error),
        'r2_score': make_scorer(r2_score)
    }
    cv_splits = 5
    cv = ShuffleSplit(n_splits=cv_splits, test_size=0.1, random_state=0)

    results_df = pd.DataFrame(columns=['idx', 'lr', 'n_estimators', 'max_depth', 'fit_time', 'score_time', 'test_root_mean_squared_error', 'test_mean_absolute_error', 'test_r2_score'])

    idx = 0
    for n_estimators, max_fea in tqdm(list(itertools.product(rf_n_estimators_space, rf_max_features))):
        rf = create_RF_model(n_estimators,max_fea)

        scores = cross_validate(rf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        
        result_dict = {}

        result_dict['idx'] = idx
        result_dict['n_estimators'] = n_estimators
        result_dict['max_feature'] = max_fea

        result_dict['fit_time'] = sum(scores['fit_time']) / cv_splits
        result_dict['score_time'] = sum(scores['score_time']) / cv_splits
        result_dict['test_root_mean_squared_error'] = np.sqrt(sum(scores['test_mean_squared_error']) / cv_splits)
        result_dict['test_mean_absolute_error'] = sum(scores['test_mean_absolute_error']) / cv_splits
        result_dict['test_r2_score'] = sum(scores['test_r2_score']) / cv_splits
        results_df = results_df.append(result_dict, ignore_index=True)
        print('IDX: {:d} | n_estimators {:.4f} | max_feature {:.4f} | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(idx, n_estimators, max_fea, result_dict['test_root_mean_squared_error'], result_dict['test_mean_absolute_error'], result_dict['test_r2_score']))
        idx += 1

    results_df.to_csv('../../out/exp2_result/rf_agb_gridsearch3.csv', index=False)
    


# grid_search('SENT+DEM+AGB', 'soc', True)
grid_search('ALL+SOC+22', 'agb', False)