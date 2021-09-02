import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define feature importance legend colour and label
blue_patch = mpatches.Patch(color='Dodgerblue', label='Sentinel 1')
org_patch = mpatches.Patch(color='orange', label='Sentinel 2')
gray_patch = mpatches.Patch(color='gray', label='Digital Elevation')
dred_patch = mpatches.Patch(color='darkred', label='LandSat 8')
green_patch = mpatches.Patch(color='darkolivegreen', label='Woodland Category')
lgreen_patch = mpatches.Patch(color='lightgreen', label='AGB')
black_patch = mpatches.Patch(color='black', label='SOC/SOCD')

# Mappings from model to features list and legend colour list
FEATURES_DICT = {
    'MODEL_A': [
        ['VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI'],
        ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3,
        [blue_patch, org_patch, gray_patch],
    ],
    'MODEL_B': [
        [ "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY"],
        ['darkred']*11 + ['darkolivegreen'],
        [dred_patch, green_patch],
    ],
    'MODEL_C': [
        ['VH_1', 'VV_1', 'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 'CATEGORY'],
        ['Dodgerblue']*2 + ['Gray'] * 4 + ['darkolivegreen'],
        [blue_patch, gray_patch, green_patch],
    ],
    'MODEL_D': [
        ['VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY"],
        ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3 + ['darkred']*11 + ['darkolivegreen'],
        [blue_patch, org_patch, gray_patch, dred_patch, green_patch],
    ],
    'MODEL_E': [
        [
        'VH_1','VV_1',
        'BAND_2','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI',
        "L_5", "L_6","L_8","L_9", "CATEGORY"
        ],
        ['Dodgerblue'] * 2 + ['orange'] * 2 + ['Gray'] * 4 + ['orange'] * 1 + ['darkred']*4 + ['darkolivegreen'],
        [blue_patch, gray_patch, org_patch, dred_patch, green_patch]
    ],
    'MODEL_F': [
        [
        'VH_1', 'VV_1', 
        'BAND_4', 'BAND_8A', 
        'DEM_CS', 'DEM_LSF', 'DEM_TWI', 'DEM_ELEV', 
        'NDVI', 
        'L_4', 'L_5', 'L_6', 'L_10', 'CATEGORY'
        ],
        ['Dodgerblue'] * 2 + ['orange'] * 2 + ['Gray'] * 4 + ['orange'] * 1 + ['darkred']*4 + ['darkolivegreen'],
        [blue_patch, gray_patch, org_patch, dred_patch, green_patch]
    ],
    'MODEL_G':[[
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF', 'DEM_TWI', 'DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "AGB"],
        ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3 + ['lightgreen'],
        [blue_patch, org_patch, gray_patch, lgreen_patch],
    ],
    'MODEL_H':[[
        'VH_1','VV_1',
        'BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS', 'DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_5", "L_6","L_7", "L_10","L_11", "CATEGORY",
        "OC", "SG_15_30"],
        ['Dodgerblue'] * 2 + ['orange'] * 7 + ['Gray'] * 2 + ['orange'] * 3 + ['darkred']*5 + ['darkolivegreen'] + 2*['black'],
        [blue_patch, org_patch, gray_patch, dred_patch, green_patch, black_patch],
    ]
}

# Generate feature graphs
def generate_feature_graph(model_name, model_path, output_path):
    
    print(FEATURES_DICT[model_name][0])
    print(FEATURES_DICT[model_name][1])
    print(FEATURES_DICT[model_name][2])

    print("Loading model", model_path, "...")
    # Load model from model path
    model = joblib.load(model_path)

    # print feature importances in model
    importances = list(model.feature_importances_)
    print("Importances:", importances)

    # Store feature importance names and values as a list
    feature_importances = [(FEATURES_DICT[model_name][0][i], list(importances)[i], FEATURES_DICT[model_name][1][i]) for i in range(len(FEATURES_DICT[model_name][0]))]
    # Sort features according to their importance
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print(len(feature_importances))

    print("Feature       Importance")
    for feature, importance, _ in feature_importances:
        print("{:<15} {:<15.2f}".format(feature, round(importance, 4)  * 100))

    feature_importances = [(fe, round(im, 4) * 100, _) for (fe, im, _) in feature_importances]    

    fh = 7
    fw = 8
    fig, ax = plt.subplots(figsize=(fh, fw))
    importances = [i for f, i, c in feature_importances]
    features = [f for f, i, c in feature_importances]
    colours = [c for f, i, c in feature_importances]
    ax.barh(features, importances, align='center', color=colours)
    ax.set_title("Relative % Importance")
    ax.set_xlabel("Relative Importance (%)")
    plt.legend(handles=FEATURES_DICT[model_name][2])
    plt.savefig(output_path)



generate_feature_graph('MODEL_A', '../../models/soc/brt_Model_A.joblib.pkl', './soc_brt_model_A_feature.png')
