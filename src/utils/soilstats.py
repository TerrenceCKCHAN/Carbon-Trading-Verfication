import numpy as np
import pandas as pd
from scipy.stats import skew
import joblib

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.02.csv"
# csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_SoilGrids2_0_roi_points_0.02.csv"
data_df = load_csv_to_pd(csv_file_path)

# model_path = "models/brtmodel_SoilGrids_nonlog_120621.joblib.pkl"
model_path = "models/votmodel_080621_1.joblib.pkl"
model = joblib.load(model_path)

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]
# features_list = [
#     'VH_1', 'VV_1', 'VH_2', 'VV_2', 'VH_3', 'VV_3', 'VH_4', 'VV_4', 'VH_5', 'VV_5',
#     'DEM_ELEV', 'DEM_CS', 'DEM_LSF', 'DEM_SLOPE', 'DEM_TWI'
# ]


X = data_df[features_list].values.astype(np.float32)

# For Models (not soilgrids)
oc_vals_log = model.predict(X)
oc_vals = np.exp(oc_vals_log)

# For TIN
# oc_vals =  data_df['OC'].values
# oc_vals_log =  np.log(data_df['OC'].values)

# For SoilGrids
# oc_vals =  data_df['OC'].str.replace('\"', '').astype(float)
# oc_vals_log =  np.log(oc_vals + 0.000001)

print("OC VALS")
print("MEAN:", np.mean(oc_vals))
print("MAX:", np.max(oc_vals))
print("MIN:", np.min(oc_vals))
print("MEDIAN:", np.median(oc_vals))
print("STD:", np.std(oc_vals))
print("SKEW:", skew(oc_vals))

print("LOG OC VALS")
print("MEAN:", np.mean(oc_vals_log))
print("MAX:", np.max(oc_vals_log))
print("MIN:", np.min(oc_vals_log))
print("MEDIAN:", np.median(oc_vals_log))
print("STD:", np.std(oc_vals_log))
print("SKEW:", skew(oc_vals_log))