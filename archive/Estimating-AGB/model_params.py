EXPANDED_SPECIES_TRAIN = ['Category_Non woodland', 'Category_Woodland',
   'Species_Assumed woodland', 'Species_Bare area', 'Species_Broadleaved',
   'Species_Conifer', 'Species_Failed', 'Species_Felled',
   'Species_Grassland', 'Species_Ground prep', 'Species_Low density',
   'Species_Mixed mainly broadleaved', 'Species_Mixed mainly conifer',
   'Species_Open water', 'Species_Other vegetation', 'Species_Quarry',
   'Species_Road', 'Species_Shrub', 'Species_Urban', 'Species_Windblow',
   'Species_Young trees']

EXPANDED_SPECIES_QUICK = ['Category_Non woodland',
    'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved',
    'Species_Conifer', 'Species_Felled', 'Species_Grassland', 'Species_Other vegetation',
    'Species_Young trees']

COMBINED_COLUMNS = ['Sentinel VH', 'Sentinel VV', 'Landsat 2', 'Landsat 3',
    'Landsat 4', 'Landsat 5', 'Landsat 6', 'Landsat 7', 'Category',
    'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI']
LANDSAT_COLUMNS= ['Landsat 2', 'Landsat 3', 'Landsat 4', 'Landsat 5',
    'Landsat 6', 'Landsat 7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI']
SENTINEL_COLUMNS = ['Sentinel VH', 'Sentinel VV']

sentinel_rf = ['Sentinel VH', 'Sentinel VV', 'Category Woodland',
    'Species Conifer', 'Species Felled', 'Species Assumed Woodland', 'Species Broadleaved',
    'Species Young Trees', 'Category Non Woodland', 'Species Agriculture', 'Species Bare Area',
    'Species Failed', 'Species Grassland', 'Species Ground Prep', 'Species Low Density']

sentinel_xgb = ['Category Woodland', 'Species Felled', 'Species Conifer', 'Category Non Woodland',
    'Species Assumed Woodland', 'Species Windblow', 'Species Broadleaved', 'Sentinel VH', 'Sentinel VV',
    'Species Grassland', 'Species Ground Prep', 'Species Mixed Mainly Broadleaved', 'Species Bare Area',
    'Species Failed', 'Species Low Density']

INVEN = ['nonwood', 'wood', 'as', 'br', 'co', 'fe', 'gr', 'ot', 'yo']
INVEN_TRAIN = ['nonwood', 'wood', 'ag', 'as', 'ba', 'br', 'co', 'fa', 'fe', 'gr',
            'gro',  'lo', 'mib', 'mic', 'op', 'ot', 'ro', 'sh', 'wib', 'yo']


xgb_params_combined = {
    'max_depth': 8,
    'min_child_weight': 4,
    'subsample': 0.6,
    'eta': 0.01,
    'gamma': 0.0,}
xgb_params_sentinel = {
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'eta': 0.05,
    'gamma': 0.0,}
xgb_params_landsat = {
    'max_depth': 6,
    'min_child_weight': 2,
    'subsample': 0.6,
    'eta': 0.01,
    'gamma': 0.4,}


rf_params_combined = {
        'n_estimators': 1400,
        'max_features': 1/6,
    }
rf_params_sentinel = {
        'n_estimators': 1400,
        'max_features': 1/6,
    }
rf_params_landsat = {
        'n_estimators': 2400,
        'max_features': 1/6,
    }


def get_columns(settings):
    dataset = settings['dataset']
    inven = settings['inven']
    column_names = []
    if dataset != 'landsat':
        column_names.extend(['Sentinel VH', 'Sentinel VV'])
    if dataset != 'sentinel':
        column_names.extend(['Landsat 2', 'Landsat 3', 'Landsat 4', 'Landsat 5', 'Landsat 6', 'Landsat 7'])
    if inven:
        column_names.extend(['Category', 'Species'])
    if dataset != 'sentinel':
        column_names.extend(['NDVI', 'SAVI', 'EVI', 'ARVI'])
    return column_names

def get_extended_columns(settings):
    column_names = get_columns(settings)
    dataset = settings['dataset']
    if dataset == 'sentinel':
        column_names = column_names[:-2]
        column_names.extend(EXPANDED_SPECIES_TRAIN)
    else:
        veg_indices = column_names[-4:]
        column_names = column_names[:(len(column_names)-6)]
        column_names.extend(EXPANDED_SPECIES_QUICK)
        column_names.extend(veg_indices)
    return column_names

def get_params(settings):
    dataset = settings['dataset']
    model = settings['model']

    if model == 'xgboost':
        if dataset == 'combined':
            return xgb_params_combined
        elif dataset == 'landsat':
            return xgb_params_landsat
        else:
            return xgb_params_sentinel
    else:
        if dataset == 'combined':
            return rf_params_combined
        elif dataset == 'landsat':
            return rf_params_landsat
        else:
            return rf_params_sentinel

def get_inven_abv(settings):
    if settings['inven']:
        if settings['dataset'] == 'sentinel' or settings['dim'] != '':
            return INVEN_TRAIN
        else:
            return INVEN
    return []
