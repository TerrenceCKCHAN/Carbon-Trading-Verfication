import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load our csv file to pandas data frame
def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

# Specify our input data csv file
csv_file_path = r"C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv"
# Get data frame object
data_df = load_csv_to_pd(csv_file_path)

# The feature list we analyse
features_list = [
    'VH_1','VV_1',
    'BAND_2','BAND_8A',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    'EVI',
    "L_5", "L_6","L_8","L_9", "CATEGORY",
    "OC"
]



df = data_df[features_list]
# Compute Peason's correlation
cor = df.corr()
# Plot heat map showing correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(cor, annot=True)
# Save plot to specified file location
plt.savefig('pearson_corr_soc_removal.png')

# Calculate VIF

# Change OC to AGB for AGB estimation
x = df.drop('OC', 1)
# Add intercept for python OLS used in statsmodel VIF calculation (https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python)
x = add_constant(x)
y = df['OC']

n = x.shape[1]
# Get variables and their corresponding VIF scores
vif = [variance_inflation_factor(x.values, i) for i in range(n)]
# Extract a list of features with vif scores less than 10
final_features = [x.columns[i] for i in range(1,n) if vif[i] <= 10]       

print("Final features list:", final_features)

