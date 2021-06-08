import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.04.csv"
lucas_csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_LUCAS2009_zhou2020_points.csv"
# csv_file_path = lucas_csv_file_path
data_df = load_csv_to_pd(csv_file_path)
lucas_data_df = load_csv_to_pd(lucas_csv_file_path)

msk = np.random.rand(len(data_df)) < 0.8
train_df = data_df[msk]
test_df = data_df[~msk]

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]

# features_list = [
#     'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI'
# ]

X_train = train_df[features_list].values.astype(np.float32)
y_train = np.log(train_df['OC'].values.astype(np.float32))

X_test = test_df[features_list].values.astype(np.float32)
y_test = np.log(test_df['OC'].values.astype(np.float32))

X_lucas = lucas_data_df[features_list].values.astype(np.float32)
y_lucas = np.log(lucas_data_df['OC'].values.astype(np.float32))

rf = RandomForestRegressor(
    n_estimators=300,
    max_features=11
).fit(X_train, y_train)

brt = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    random_state=0,
    loss='ls'
).fit(X_train, y_train)

vot = VotingRegressor([('rf', rf), ('brt', brt)]).fit(X_train, y_train)

test_rmse = np.sqrt(mean_squared_error(y_test, vot.predict(X_test)))
test_mae = mean_absolute_error(y_test, vot.predict(X_test))
test_r2 = r2_score(y_test, vot.predict(X_test))
print('TEST RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(test_rmse, test_mae, test_r2))

lucas_rmse = np.sqrt(mean_squared_error(y_lucas, vot.predict(X_lucas)))
lucas_mae = mean_absolute_error(y_lucas, vot.predict(X_lucas))
lucas_r2 = r2_score(y_lucas, vot.predict(X_lucas))
print('LUCAS ZHOU2020 RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(lucas_rmse, lucas_mae, lucas_r2))

joblib.dump(vot, "models/votmodel.joblib.pkl", compress=3)