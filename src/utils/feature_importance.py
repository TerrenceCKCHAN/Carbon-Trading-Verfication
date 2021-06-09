import joblib

features_list = [
    'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
    'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
    'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
]

model_path = "models/brtmodel_SoilGrids_090621_1.joblib.pkl"
# model_path = "models/rfmodel_080621_1.joblib.pkl"
print("Loading model", model_path, "...")
model = joblib.load(model_path)

importances = list(model.feature_importances_)
print("Importances:", importances)

feature_importances = [(feature, importance) for feature, importance in zip(features_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("Feature       Importance%")
for feature, importance in feature_importances:
    print("{:<15} {:<15.2f}".format(feature, round(importance, 4)  * 100))
