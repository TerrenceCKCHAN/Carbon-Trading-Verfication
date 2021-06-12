import numpy as np
import pandas as pd
from scipy.stats import skew

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

# csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.02.csv"
csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_SoilGrids2_0_roi_points_0.02.csv"
data_df = load_csv_to_pd(csv_file_path)

# oc_vals =  data_df['OC'].values
# oc_vals_log =  np.log(data_df['OC'].values)

oc_vals =  data_df['OC'].str.replace('\"', '').astype(float)
oc_vals_log =  np.log(oc_vals + 0.000001)

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