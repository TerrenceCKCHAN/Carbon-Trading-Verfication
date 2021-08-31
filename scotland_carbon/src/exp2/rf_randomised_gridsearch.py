import numpy as np
import pandas as pd
import joblib
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, ShuffleSplit
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV


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
    Y = np.log(data_df[ground_truth_col].values.astype(np.float32)) if isLog else data_df[ground_truth_col].values.astype(np.float32)

    rf = RandomForestRegressor()
    random_grid = {
        'max_features': ['auto', 'sqrt'],
        'n_estimators': [x for x in range(100,3000,100)]

    }
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X, Y)
    print(rf_random.best_params_)

grid_search('ALL+SOC+22', 'agb', False)