import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

# csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.04.csv"
csv_file_path = r"C:\Users\kothi\Documents\individual_project\individual_project\data\S1AIW_S2AL2A_NDVI_EVI_SATVI_DEM_LUCASTIN_roi_points_0.02.csv"
data_df = load_csv_to_pd(csv_file_path)

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI',
    'OC'
]

df = data_df[features_list]

# Calculate correlation between variables
cor = df.corr()
print(cor)

plt.figure(figsize=(10,6))
sns.heatmap(cor, annot=True)
plt.show()

# Calculate VIF

x = df.drop('OC', 1)
# Need to add constant to match calculation in R (https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python)
x = add_constant(x)
y = df['OC']

thresh = 10
final_feature_list = []

n = x.shape[1]

vif = [variance_inflation_factor(x.values, i) for i in range(n)]
print(pd.Series(vif, index=x.columns))
for i in range(1, n):
    if vif[i] <= thresh:
        final_feature_list.append(x.columns[i])

print("Final features list:", final_feature_list)