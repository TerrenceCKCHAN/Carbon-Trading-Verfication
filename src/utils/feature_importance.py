import joblib
import matplotlib.pyplot as plt

# features_list = [
#     'VH_1','VV_1','VH_2','VV_2','VH_3','VV_3','VH_4','VV_4','VH_5','VV_5',
#     'BAND_11','BAND_12','BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_8','BAND_8A','NDVI','EVI','SATVI',
#     'DEM_ELEV','DEM_CS','DEM_LSF','DEM_SLOPE','DEM_TWI'
# ]

# colour_list = [
#     'r','r','r','r','r','r','r','r','r','r',
#     'g','g','g','g','g','g','g','g','g','g','g','g','g',
#     'b','b','b','b','b'
# ]

features_list = [
    'VH_1', 'VV_1', 'VH_2', 'VV_2', 'VH_3', 'VV_3', 'VH_4', 'VV_4', 'VH_5', 'VV_5',
    'DEM_ELEV', 'DEM_CS', 'DEM_LSF', 'DEM_SLOPE', 'DEM_TWI'
]

colour_list = [
    'r','r','r','r','r','r','r','r','r','r',
    'b','b','b','b','b'
]

# model_path = "models/rfmodel_080621_1.joblib.pkl"
model_path = "models/rfmodel.joblib.pkl"
print("Loading model", model_path, "...")
model = joblib.load(model_path)

importances = list(model.feature_importances_)
print("Importances:", importances)

feature_importances = [(features_list[i], list(importances)[i], colour_list[i]) for i in range(len(features_list))]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(len(feature_importances))

print("Feature       Importance%")
for feature, importance, _ in feature_importances:
    print("{:<15} {:<15.2f}".format(feature, round(importance, 4)  * 100))

fig, ax = plt.subplots(figsize=(8, 10))
importances = [i for f, i, c in feature_importances]
features = [f for f, i, c in feature_importances]
colours = [c for f, i, c in feature_importances]
ax.barh(features, importances, align='center', color=colours)
ax.set_title("Relative % Importance (RF Non Collinear)")
ax.set_xlabel("Relative Importance (%)")
ax.set_ylabel("Feature")

plt.show()