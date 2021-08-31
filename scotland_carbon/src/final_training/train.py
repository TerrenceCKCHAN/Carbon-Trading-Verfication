import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost

csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"

MODEL_DICT = {
    'brt': GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        random_state=0,
        loss='ls'),
    'xgb': xgboost.XGBRegressor(
        min_child_weight=2,
        n_estimators=451,
        max_depth=6,
        eta=0.01,
        subsample=0.6),
    'rf': RandomForestRegressor(
        n_estimators=300,
        max_features=7
    )
}

FEATURES_DICT = {
    'MODEL_D': [
    'VH_1','VV_1',
    'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    'EVI', 'NDVI','SATVI',
    "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY"
    ],
    'MODEL_E': [
        'VH_1','VV_1',
        'BAND_2','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI',
        "L_5", "L_6","L_8","L_9", "CATEGORY",
        "AGB"
    ],
    'MODEL_F': [
        'VH_1', 'VV_1', 
        'BAND_4', 'BAND_8A', 
        'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 
        'NDVI', 
        'L_4', 'L_5', 'L_6', 'L_10', 
        'CATEGORY',
        'OC', 'SG_15_30'
    ],
    'MODEL_G': [
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        'AGB'
    ],
    'MODEL_H': [
        'VH_1','VV_1',
        'BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_5", "L_6","L_7","L_10","L_11", "CATEGORY",
        "OC", "SG_15_30"
    ],
}

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

def train(input_csv, feature, pred, model, log):

    ground_truth = 'AGB' if pred == 'agb' else 'OC'

    data_df = load_csv_to_pd(input_csv)
    msk = np.random.rand(len(data_df)) < 0.8
    train_df = data_df[msk]
    test_df = data_df[~msk]
    X_train = train_df[FEATURES_DICT[feature]].values.astype(np.float32)
    Y_train = np.log(train_df[ground_truth].values) if log else train_df[ground_truth].values.astype(np.float32)
    X_test = test_df[FEATURES_DICT[feature]].values.astype(np.float32)
    Y_test = np.log(test_df[ground_truth].values) if log else test_df[ground_truth].values.astype(np.float32)

    m = MODEL_DICT[model].fit(X_train,Y_train)

    test_rmse = np.sqrt(mean_squared_error(Y_test, m.predict(X_test)))
    test_mae = mean_absolute_error(Y_test, m.predict(X_test))
    test_r2 = r2_score(Y_test, m.predict(X_test))
    result_string = 'TEST RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(test_rmse, test_mae, test_r2)
    print(result_string)

    feature2model = {
        'ALL+AGB': 'all+agb_',
        'ALL+SOC': 'all+soc_',
        'SENT+DEM+AGB': 'sent+dem+agb_',
        'ALL+SOC+22': 'all+soc+22_',
        'VIF+AGB': 'vif+agb_',
        'VIF+SOC': 'vif+soc_'
    }
    out_name = '../../models/exp3_models/' + pred + '/' + model + '_' + feature2model[feature] + 'model.joblib.pkl'
    out_result = '../../models/exp3_models/' + pred + '/' + model + '_' + feature2model[feature] + 'result.txt'
    
    joblib.dump(m, out_name, compress=3)

    with open(out_result, "w") as text_file:
        text_file.write(result_string)

# train - (input file, features to use, predicting for agb/soc, model used (brt, rf, xgb), log output or not)
print("Training model 1")

train(csv_file_path, 'SENT+DEM+AGB', 'soc', 'rf', True)


# train(csv_file_path, 'ALL+SOC+22', 'agb', 'xgb', False)

# train(csv_file_path, 'VIF+AGB', 'soc', 'brt', True)
# train(csv_file_path, 'VIF+AGB', 'soc', 'rf', True)
# train(csv_file_path, 'VIF+AGB', 'soc', 'xgb', True)

# train(csv_file_path, 'VIF+SOC', 'agb', 'brt', False)
# train(csv_file_path, 'VIF+SOC', 'agb', 'rf', False)
# train(csv_file_path, 'VIF+SOC', 'agb', 'xgb', False)

# train(csv_file_path, 'VIF', 'soc', 'brt', True)
# train(csv_file_path, 'VIF', 'soc', 'rf', True)
# train(csv_file_path, 'VIF', 'soc', 'xgb', True)

# train(csv_file_path, 'VIF', 'agb', 'brt', False)
# train(csv_file_path, 'VIF', 'agb', 'rf', False)
# train(csv_file_path, 'VIF', 'agb', 'xgb', False)

# train(csv_file_path, 'ALL', 'soc', 'brt', True)
# train(csv_file_path, 'ALL', 'soc', 'rf', True)
# train(csv_file_path, 'ALL', 'soc', 'xgb', True)

# train(csv_file_path, 'ALL', 'agb', 'brt', False)
# train(csv_file_path, 'ALL', 'agb', 'rf', False)
# train(csv_file_path, 'ALL', 'agb', 'xgb', False)
