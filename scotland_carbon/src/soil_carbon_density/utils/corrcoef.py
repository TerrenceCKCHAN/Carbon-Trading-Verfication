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
    'BAND_2','BAND_8A',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    'EVI',
    "L_5", "L_6","L_8","L_9", "CATEGORY",
    "OC"
]

# Remove Band 3, L1, band 12, L11, B7, B11, B6, L2, L3, B4, B5, SATVI, L4, L7, L10, EVI

# pd.options.display.float_format = '{:,.0f}'.format

df = data_df[features_list]

# Calculate correlation between variables
cor = df.corr()
print(cor)

plt.figure(figsize=(20,20))
sns.heatmap(cor, annot=True)
plt.savefig('../../../report_output/experiment1/pearson_corr_soc_removal.png')
# plt.show()

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

