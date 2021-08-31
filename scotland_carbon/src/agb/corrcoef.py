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

csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"
data_df = load_csv_to_pd(csv_file_path)

features_list = [
    'VH_1','VV_1',
    'BAND_4','BAND_8A',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    'NDVI',
    "L_4", "L_5", "L_6", "L_10", "CATEGORY",
    "AGB"
]

# BAND_12, BAND_6, BAND_7, BAND_2, Band_11, BAND_5, band 4, L2, L11,L_9, L_3, L_8, L_7, L_1,SATVI
# Band 3,4 highly correlated

# pd.options.display.float_format = '{:,.0f}'.format

df = data_df[features_list]

# Calculate correlation between variables
cor = df.corr()
print(cor)

plt.figure(figsize=(20,20))
sns.heatmap(cor, annot=True)
plt.savefig('../../report_output/experiment1/pearson_corr_agb_removal.png')
# plt.show()

# Calculate VIF

x = df.drop('AGB', 1)
# Need to add constant to match calculation in R (https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python)
x = add_constant(x)
y = df['AGB']

thresh = 10
final_feature_list = []

n = x.shape[1]

vif = [variance_inflation_factor(x.values, i) for i in range(n)]
print(pd.Series(vif, index=x.columns))
for i in range(1, n):
    if vif[i] <= thresh:
        final_feature_list.append(x.columns[i])

print("Final features list:", final_feature_list)

# ['VH_1', 'VV_1', 'BAND_4', 'BAND_8A', 'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 'NDVI', 'L_4', 'L_5', 'L_6', 'L_10', 'CATEGORY']