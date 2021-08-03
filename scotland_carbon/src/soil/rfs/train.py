import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"
data_df = load_csv_to_pd(csv_file_path)

msk = np.random.rand(len(data_df)) < 0.8
train_df = data_df[msk]
test_df = data_df[~msk]

# features_list = [
#     'VH_1', 'VV_1', 
#     'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8A',
#     'DEM_ELEV', 'DEM_TWI'
# ]

features_list = [
    'VH_1','VV_1',
    'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    'EVI', 'NDVI','SATVI'
]

X_train = train_df[features_list].values.astype(np.float32)
y_train = np.log(train_df['SG_15_30'].values)
# y_train = train_df['SG_15_30'].values.astype(np.float32)


X_test = test_df[features_list].values.astype(np.float32)
y_test = np.log(test_df['SG_15_30'].values)
# y_test = test_df['SG_15_30'].values.astype(np.float32)

rf = RandomForestRegressor(
    n_estimators=300,
    max_features=11
).fit(X_train, y_train)

test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
test_mae = mean_absolute_error(y_test, rf.predict(X_test))
test_r2 = r2_score(y_test, rf.predict(X_test))
print('TEST RESULTS: | RMSE {:.4f} | MAE {:.4f} | R2 {:.4f}'.format(test_rmse, test_mae, test_r2))

joblib.dump(rf, "../../models/rfmodel.joblib.pkl", compress=3)