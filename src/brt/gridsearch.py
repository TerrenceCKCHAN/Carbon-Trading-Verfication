import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, ShuffleSplit
from tqdm import tqdm
from sklearn.metrics import make_scorer

n_estimators_space = [1, 10, 50, 100, 200, 500, 1000]
lr_space = [0.5, 0.1, 0.05, 0.01, 0.005]
max_depth_space = [1, 2, 3, 4, 5, 10]


def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.04.csv"
data_df = load_csv_to_pd(csv_file_path)

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]

X = data_df[features_list].values.astype(np.float32)
y = np.log(data_df['OC'].values.astype(np.float32))

scoring = {
    'mean_squared_error': make_scorer(mean_squared_error), 
    'mean_absolute_error': make_scorer(mean_squared_error),
    'r2_score': make_scorer(r2_score)
}
cv_splits = 5
cv = ShuffleSplit(n_splits=cv_splits, test_size=0.2, random_state=0)

results_df = pd.DataFrame(columns=['idx', 'lr', 'n_estimators', 'max_depth', 'fit_time', 'score_time', 'test_root_mean_squared_error', 'test_mean_absolute_error', 'test_r2_score'])

idx = 0
for lr, n_estimators, max_depth in tqdm(list(itertools.product(lr_space, n_estimators_space, max_depth_space))):
    brt = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=max_depth,
        random_state=0,
        loss='ls'
    )

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

results_df.to_csv('out/brt_gridsearch.csv', index=False)