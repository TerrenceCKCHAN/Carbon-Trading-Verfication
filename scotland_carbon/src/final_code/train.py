import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost
from utils import load_csv_to_pd, FEATURES_DICT

# Specify input data location
csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"

# Dictionary storing mapping from machine learning name to machine learning model
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

# Map target variable names to column names in input csv file
PRED_DICT = {
    'agb': 'AGB',
    'soc': 'OC',
    'socd': 'SG_15_30'
}

def train(feature, pred, model, log, model_path):
    # Get the ground truth label for our target variable
    ground_truth = PRED_DICT[pred]
    # load data
    data_df = load_csv_to_pd(csv_file_path)
    # Create train test data set
    mask = np.random.rand(len(data_df)) < 0.8
    train_df = data_df[mask]
    test_df = data_df[~mask]
    # Get training data from feature list
    X_train = train_df[FEATURES_DICT[feature]].values.astype(np.float32)
    # Get training ground truth data, log for SOC predictions, no log for AGB predictions
    Y_train = np.log(train_df[ground_truth].values) if log else train_df[ground_truth].values.astype(np.float32)
    # Get test data from feature list
    X_test = test_df[FEATURES_DICT[feature]].values.astype(np.float32)
    # Get testing ground truth data
    Y_test = np.log(test_df[ground_truth].values) if log else test_df[ground_truth].values.astype(np.float32)

    # Get machine learning model and train on xtrain and ytrain data
    m = MODEL_DICT[model].fit(X_train,Y_train)

    # Get Evaluation metrics - RMSE, MAE, R2
    test_rmse = np.sqrt(mean_squared_error(Y_test, m.predict(X_test)))
    test_mae = mean_absolute_error(Y_test, m.predict(X_test))
    test_r2 = r2_score(Y_test, m.predict(X_test))

    # Format evluation metrics into result string and print out
    result = 'TEST RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(test_rmse, test_mae, test_r2)
    print(result)

    # Construct output model file name and output result file name
    out_name   = model_path + '/' + pred + '_' + model + '_' + feature + '_' + '.joblib.pkl'
    out_result = model_path + '/' + pred + '_' + model + '_' + feature + '_' + 'result.txt'
    
    # Store model
    joblib.dump(m, out_name, compress=3)
    # Store result
    with open(out_result, "w") as text_file:
        text_file.write(result)

# Train parameters:
# (feature model, target variable (soc,agb,socd) , machine learning technique (brt, rf, xgb), boolean log or not, output path) 
# train - (features to use, predicting for agb/soc, model used (brt, rf, xgb), log output or not)
train('MODEL_A', 'soc', 'rf', True, '.')
