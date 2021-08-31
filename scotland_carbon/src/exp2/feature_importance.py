import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

blue_patch = mpatches.Patch(color='Dodgerblue', label='Sentinel 1')
org_patch = mpatches.Patch(color='orange', label='Sentinel 2')
gray_patch = mpatches.Patch(color='gray', label='Digital Elevation')
dorg_patch = mpatches.Patch(color='darkorange', label='LandSat 8')
green_patch = mpatches.Patch(color='darkolivegreen', label='Woodland Category')
lgreen_patch = mpatches.Patch(color='lightgreen', label='AGB')
black_patch = mpatches.Patch(color='black', label='SOC/SOCD')

FEATURES_DICT = {
    'ALL+AGB': [[
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY",
        "AGB"],
        ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3 + ['darkorange']*11 + ['darkolivegreen'] + ['lightgreen'],
        [blue_patch, org_patch, gray_patch, dorg_patch, green_patch, lgreen_patch],
    ],
    'ALL+SOC': [[
        'VH_1','VV_1',
        'BAND_2','BAND_3','BAND_4','BAND_5','BAND_6','BAND_7','BAND_11','BAND_12','BAND_8A',
        'DEM_CS','DEM_LSF','DEM_TWI','DEM_ELEV',
        'EVI', 'NDVI','SATVI',
        "L_1","L_2","L_3","L_4", "L_5", "L_6","L_7","L_8","L_9","L_10","L_11", "CATEGORY",
        "OC", "SG_15_30"],
        ['Dodgerblue'] * 2 + ['orange'] * 9 + ['Gray'] * 4 + ['orange'] * 3 + ['darkorange']*11 + ['darkolivegreen'] + 2*['black'],
        [blue_patch, org_patch, gray_patch, dorg_patch, green_patch, black_patch],
    ]
}

feature2model = {
    'ALL+SOC': 'all+soc_',
    'ALL+AGB': 'all+agb_'
}

def generate_feature_graph(f, pred, mod):
    print(FEATURES_DICT[f][0])
    print(FEATURES_DICT[f][1])
    print(FEATURES_DICT[f][2])

    
    model_path = '../../models/exp2_models/' + pred + '/' + mod + '_' + feature2model[f] + 'model.joblib.pkl'
    print("Loading model", model_path, "...")
    model = joblib.load(model_path)

    importances = list(model.feature_importances_)
    print("Importances:", importances)

    feature_importances = [(FEATURES_DICT[f][0][i], list(importances)[i], FEATURES_DICT[f][1][i]) for i in range(len(FEATURES_DICT[f][0]))]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print(len(feature_importances))

    print("Feature       Importance")
    for feature, importance, _ in feature_importances:
        print("{:<15} {:<15.2f}".format(feature, round(importance, 4)  * 100))

    feature_importances = [(fe, round(im, 4) * 100, _) for (fe, im, _) in feature_importances]    

    fh = 7
    fw = 8 if f == 'ALL' else 4
    fig, ax = plt.subplots(figsize=(fh, fw))
    importances = [i for f, i, c in feature_importances]
    features = [f for f, i, c in feature_importances]
    colours = [c for f, i, c in feature_importances]
    ax.barh(features, importances, align='center', color=colours)
    ax.set_title("Relative % Importance")
    ax.set_xlabel("Relative Importance (%)")
    # ax.set_ylabel("Feature")
    plt.legend(handles=FEATURES_DICT[f][2])
    plt.savefig('../../report_output/exp2/feature_maps/' + pred +'/' + mod + '_' + feature2model[f] + 'feature.png')

# generate_feature_graph('ALL+AGB', 'soc', 'brt')
# generate_feature_graph('ALL+AGB', 'soc', 'rf')
# generate_feature_graph('ALL+AGB', 'soc', 'xgb')

generate_feature_graph('ALL+SOC', 'agb', 'brt')
generate_feature_graph('ALL+SOC', 'agb', 'rf')
generate_feature_graph('ALL+SOC', 'agb', 'xgb')
