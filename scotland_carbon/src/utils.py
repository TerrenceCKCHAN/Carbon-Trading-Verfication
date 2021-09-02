
import pandas as pd
# Load csv file to pandas dataframe
def load_csv_to_pd(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=r'\s*,\s*', engine='python')
    df.drop_duplicates(subset=None, inplace=True)
    return df

# Dictionary mapping model names to the corresponding list of features
FEATURES_DICT = {
    'MODEL_A': [
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI'
    ],
    'MODEL_B': [
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY", 
    ],
    'MODEL_C': ['VH_1', 'VV_1', 'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 'CATEGORY'],
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
    ],
    'MODEL_F': [
        'VH_1', 'VV_1', 
        'BAND_4', 'BAND_8A', 
        'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 
        'NDVI', 
        'L_4', 'L_5', 'L_6', 'L_10', 
        'CATEGORY',
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