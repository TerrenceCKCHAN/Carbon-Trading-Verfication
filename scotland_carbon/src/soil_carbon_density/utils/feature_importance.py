import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# features_list = [
#     'VH_1','VV_1',
#     'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
#     'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
#     'EVI', 'NDVI','SATVI',
# ]

# colour_list = ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3

# features_list = ['VH_1', 'VV_1', 'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 'CATEGORY']
# colour_list = ['Dodgerblue'] * 2 + ['Gray'] * 4 + ['darkolivegreen']

features_list = [
    'VH_1','VV_1',
    'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
    'EVI', 'NDVI','SATVI',
    'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
    "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY"
]
colour_list = ['Dodgerblue'] * 2 + ['orange'] * 12 + ['Gray'] * 4 + ['darkred']*11 + ['darkolivegreen']




# model_path = "models/rfmodel_080621_1.joblib.pkl"
# model_path = "models/brtmodel_SoilGrids_nonlog_120621.joblib.pkl"
model_path = "../../../models/soil_carbon_density_models/brt_all_model.joblib.pkl"
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

feature_importances = [(fe, round(im, 4) * 100, _) for (fe, im, _) in feature_importances]   

# fig, ax = plt.subplots(figsize=(7, 4))


fig, ax = plt.subplots(figsize=(7, 8))
importances = [i for f, i, c in feature_importances]
features = [f for f, i, c in feature_importances]
colours = [c for f, i, c in feature_importances]
ax.barh(features, importances, align='center', color=colours)
ax.set_title("Relative % Importance")
ax.set_xlabel("Relative Importance (%)")
# ax.set_ylabel("Feature")


# blue_patch = mpatches.Patch(color='Dodgerblue', label='Sentinel 1')
# orange_patch = mpatches.Patch(color='orange', label='Sentinel 2')
# green_patch = mpatches.Patch(color='Gray', label='DEM Derivatives')


# blue_patch = mpatches.Patch(color='Dodgerblue', label='Sentinel 1')
# orange_patch = mpatches.Patch(color='Gray', label='DEM Derivatives')
# green_patch = mpatches.Patch(color='darkolivegreen', label='WoodLand Category')

blue_patch = mpatches.Patch(color='Dodgerblue', label='Sentinel 1')
org_patch = mpatches.Patch(color='orange', label='Sentinel 2')
gray_patch = mpatches.Patch(color='gray', label='Digital Elevation')
dred_patch = mpatches.Patch(color='darkred', label='LandSat 8')
green_patch = mpatches.Patch(color='darkolivegreen', label='Woodland Category')

# plt.legend(handles=[blue_patch, orange_patch,green_patch])
plt.legend(handles=[blue_patch, org_patch, gray_patch, dred_patch, green_patch])
plt.savefig('../../../report_output/experiment1/brt_all_feature_im.png')
# plt.show()