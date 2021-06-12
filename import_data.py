#from osgeo import gdal
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from os import path as op
import pickle

import geopandas as gpd
import shapely as shp
import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats
from PIL import Image
from skimage import io


# 8 values of carbon recorded:
    # Carbon Standing (tCO2e/ ha/yr)
    # Debris (tCO2e/ha/yr)
    # Total (tCO2/ha/yr)
    # Cumulative in-period (tCO2e/ha/ 5yr period)
    # Cum. Biomass Sequestrn (tCO2e/ha)
    # Cum. Emis. Ongoing Mgmt (tCO2e/ha)
    # Cumulative Total Sequestrn (tCO2e/ha)
    # Removed from Forest (tCO2e/ha/yr)
CARBON_MEASUREMENTS = [6, 7, 8, 9, 10, 11, 12, 13]

species_dict = {
    'Broadleaved':0,
    'Conifer': 1,
    'Felled': 2,
    'Ground prep': 3,
    'Mixed mainly broadleaved': 4,
    'Mixed mainly conifer': 5,
    'Young trees': 6,
    'Shrub': 7,
    'Low density': 8,
    'Assumed woodland': 9,
    'Open water': 10,
    'Grassland': 11,
    'Agriculture land': 12,
    'Urban': 13,
    'Road':14,
    'Quarry': 15,
    'Bare area': 16,
    'Windfarm': 17,
    'Windblow': 18,
    'Other vegetation': 19,
    'Failed':20
}

TRAIN_COLUMNS = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4',
   'Landsat5', 'Landsat6', 'Landsat7',
   'Category_Non woodland', 'Category_Woodland',
   'Species_Assumed woodland', 'Species_Bare area', 'Species_Broadleaved',
   'Species_Conifer', 'Species_Failed', 'Species_Felled',
   'Species_Grassland', 'Species_Ground prep', 'Species_Low density',
   'Species_Mixed mainly broadleaved', 'Species_Mixed mainly conifer',
   'Species_Open water', 'Species_Other vegetation', 'Species_Quarry',
   'Species_Road', 'Species_Shrub', 'Species_Urban', 'Species_Windblow',
   'Species_Young trees', 'NDVI', 'SAVI', 'EVI', 'ARVI']

TRAIN_COLUMNS_SENTINEL = ['Sentinel B1', 'Sentinel B2',
    'Category_Non woodland', 'Category_Woodland',
    'Species_Assumed woodland', 'Species_Bare area', 'Species_Broadleaved',
    'Species_Conifer', 'Species_Failed', 'Species_Felled',
    'Species_Grassland', 'Species_Ground prep', 'Species_Low density',
    'Species_Mixed mainly broadleaved', 'Species_Mixed mainly conifer',
    'Species_Open water', 'Species_Other vegetation', 'Species_Quarry',
    'Species_Road', 'Species_Shrub', 'Species_Urban', 'Species_Windblow',
    'Species_Young trees', 'NDVI', 'SAVI', 'EVI', 'ARVI']

TRAIN_COLUMNS_LANDSAT = ['Landsat2', 'Landsat3', 'Landsat4',
   'Landsat5', 'Landsat6', 'Landsat7',
   'Category_Non woodland', 'Category_Woodland',
   'Species_Assumed woodland', 'Species_Bare area', 'Species_Broadleaved',
   'Species_Conifer', 'Species_Failed', 'Species_Felled',
   'Species_Grassland', 'Species_Ground prep', 'Species_Low density',
   'Species_Mixed mainly broadleaved', 'Species_Mixed mainly conifer',
   'Species_Open water', 'Species_Other vegetation', 'Species_Quarry',
   'Species_Road', 'Species_Shrub', 'Species_Urban', 'Species_Windblow',
   'Species_Young trees', 'NDVI', 'SAVI', 'EVI', 'ARVI']

OTHER_COLUMNS = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3',
    'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category_Non woodland',
    'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved',
    'Species_Conifer', 'Species_Felled', 'Species_Grassland',
    'Species_Young trees', 'Species_Other vegetation', 'NDVI',
    'SAVI', 'EVI', 'ARVI']

OTHER_COLUMNS_SENTINEL = ['Sentinel B1', 'Sentinel B2', 'Category_Non woodland',
    'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved',
    'Species_Conifer', 'Species_Felled', 'Species_Grassland',
    'Species_Young trees', 'Species_Other vegetation', 'NDVI',
    'SAVI', 'EVI', 'ARVI']

OTHER_COLUMNS_LANDSAT = ['Landsat2', 'Landsat3',
    'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category_Non woodland',
    'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved',
    'Species_Conifer', 'Species_Felled', 'Species_Grassland',
    'Species_Young trees', 'Species_Other vegetation', 'NDVI',
    'SAVI', 'EVI', 'ARVI']

# Classes with less than 5 instances
NOT_INCLUDE = ['Uncertain', 'Coppice', 'Power line', 'River']


def not_empty_entries(row, indeces):
    for index in indeces:
        if row[index] == None or row[index] == '':
            return False
    return True

def check_possible(line_count, row, empty_indices, species_index, carbon_index):
    return line_count > 0 and not_empty_entries(row, empty_indices) and row[species_index] not in NOT_INCLUDE and float(row[carbon_index]) < 65500

def find_dict_num(forest_list, category, species):
    forest_tuple = (category, species)
    if forest_tuple not in forest_list:
        forest_list.append(forest_tuple)
    return forest_list, forest_list.index(forest_tuple)


def import_satellite_data():
    data = []
    labels = []

    with open('data/satellite_carbon_global.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if check_possible(line_count, row, [12,13, 14], 4, 14):
                forest_list, forest_id = find_dict_num(forest_list, row[3], row[4])
                # 0 fid, 1 OBJECTID_1, 2 OBJECTID,
                # 3 CATEGORY, 4 IFT_IOA, 5 COUNTRY,
                # 6 Shape_Leng, 7 Area_ha, 8 Shape__Are,
                # 9 Shape__Len, 10 x_coord, 11 y_coord,
                # 12 satellite1, 13 satellite2, 14 carbon1
                #print(f'appending {forest_id} for {row[3], row[4]} tuple')
                category = 0
                if row[3] == 'Non woodland':
                    category = 1
                data.append([category, row[4], float(row[6]), float(row[8]), float(row[10]), float(row[11]), float(row[12]), float(row[13])])
                # Carbon
                labels.append([float(row[14])])

            line_count += 1

    data = DataFrame(data, columns = ['Category', 'Species', 'Shape_Leng', 'Shape__Are', 'x_coord', 'y_coord', 'satellite1', 'satellite2'])
    labels = DataFrame(labels, columns = ['Carbon'])
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

    #print(data)
    #print(labels)
    #return data

# def update_average(running_avg, row, year_weight):
#     for i in range(len(CARBON_MEASUREMENTS)):
#         if not_empty_entry(row, i+6): # Row index of interest starts at 6
#             running_avg[i] += float(row[i + 6]) * year_weight
#     return running_avg

def import_tree_carbon_data():
    species_train = []
    species_test = []
    carbon_train = []
    carbon_test = []

    with open('data/carbon_lookup.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # year_weight = 1
        # running_avg = [0, 0, 0, 0, 0, 0, 0, 0]
        line_count = -1
        for row in csv_reader:
            if line_count > 0 and not_empty_entries(row, CARBON_MEASUREMENTS):
                if row[5] != '195-200':
                    running_avg= update_average(running_avg, row, year_weight)
                    year_weight += 1
                else:
                    species.append([row[1], row[2], row[3], row[4]]) # Species, Spacing (m), Yield Class, Management
                    carbon.append([running_avg])
                    year_weight = 1 # Reset
            line_count += 1

    species = np.asarray(species)
    carbon = np.asarray(carbon)

    print(f'Processed {line_count} lines.')
    print(f'Number of species {len(species)}.')

    return species, carbon

def import_satellite_raster(file):
    scotland_satellite_dir = 'data/scottish_subset_british_crs.tif'

    scotland_satellite = io.imread(file)
    # scotland_satellite = Image.open(scotland_satellite_dir)
    #print(scotland_satellite.shape)
    #print(scotland_satellite)
    satellite_pairs = scotland_satellite.T
    print(np.nonzero(scotland_satellite))
    #print(satellite_pairs)
    non_zero = scotland_satellite[:,221:14936,:]
    print(non_zero)
    print(non_zero.shape)
    print(np.nonzero(non_zero))

    return satellite_pairs

def import_merged_raster():
    scotland_carbon_dir = 'data/region_combined_satellite_carbon.tif'
    scotland_carbon = io.imread(scotland_carbon_dir)
    print(scotland_carbon.shape)
    print(scotland_carbon)

    return scotland_carbon

def import_grid(file):
    data = []
    labels = []

    with open('data/grid_carbon_inv_'+ file + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        empty_inventory = 0
        for row in csv_reader:
            # id,left,top,right,bottom,carbon1,satellite1,satellite2,CATEGORY_2,IFT_IOA,Shape_Leng,Shape__Are
            # Ensure it is not the first row and there are satellite readings
            if check_possible(line_count, row, [6,7], 9, 5):
                if not not_empty_entries(row, [8,9,5]):
                    empty_inventory += 1
                    row[8] = "Unknown"
                    row[9] = "Unknown"
                    row[10] = 0
                    row[11] = 0
                # CATEGORY, IFT_IOA, Shape_Leng, Shape__Are, x_coord, y_coord, satellite1, satellite2
                #print(f'appending {forest_id} for {row[3], row[4]} tuple')
                data.append([row[8], row[9], float(row[10]), float(row[11]), float(row[6]), float(row[7])])
                # Carbon
                labels.append([float(row[5])])

            line_count += 1

    data = DataFrame(data, columns = ['Category', 'Species', 'Shape_Leng', 'Shape__Are', 'satellite1', 'satellite2'])
    labels = DataFrame(labels, columns = ['Carbon'])
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    print(f'Empty inventory entries {empty_inventory}')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

def import_data():
    data_grid, labels_grid = import_grid('fromGlobal')
    print(f'Grid data size {len(data_grid.index)}')
    data, labels = import_satellite_data()
    print(f'CSV size {len(data.index)}')
    all_data = [data_grid, data]
    all_labels = [labels_grid, labels]

    data = pd.concat(all_data)
    labels = pd.concat(all_labels)
    print(f'Data size {len(data.index)}')


    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)

    return data, labels

def import_small_region(file):
    data = []
    labels = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1:
                # 0 fid 1 id 2 species1 3 sentinel1 4 sentinel2 5 carbon1
                # 6 landsat1  7 landsat2 8 landsat3 9 landsat4
                # 10 landsat5 11 landsat6 12 landsat7
                # 13 CATEGORY, 14 IFT_IOA, 15 Shape__Are, 16 Shape__Len, 17 x_coord, 18 y_coord

                band2 = float(row[7])
                band4 = float(row[9])
                band5 = float(row[10])

                ndvi = (band5 - band4) / (band5 + band4)

                savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

                data.append([float(row[3]), float(row[4]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]),  float(row[12]), row[13],  row[14], ndvi, savi, evi, arvi])
                # Carbon
                labels.append([float(row[5])])

            line_count += 1

    data = DataFrame(data, columns = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    column_names = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category_Non woodland', 'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved', 'Species_Conifer', 'Species_Felled', 'Species_Grassland', 'Species_Young trees', 'Species_Other vegetation', 'NDVI', 'SAVI', 'EVI', 'ARVI']
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)

    #data.to_csv('data/small_region/grid_output.csv')
    #print(data.head(2))
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

    #print(data)
    #print(labels)
    #return data

def import_medium_region(file):
    data = []
    labels = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1:

                # 0 fid 1 CATEGORY 2 species1 3 sentinel1 4 sentinel2
                # 5 landsat1  6 landsat2 7 landsat3 8 landsat4
                # 9 landsat5 10 landsat6 11 landsat7
                # 12 carbon, 13 species

                band2 = float(row[6])
                band4 = float(row[8])
                band5 = float(row[9])

                ndvi = (band5 - band4) / (band5 + band4)

                savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

                data.append([float(row[3]), float(row[4]), float(row[6]), float(row[7]), float(row[8]), float(row[9]),  float(row[10]), float(row[11]), row[1],  row[2], ndvi, savi, evi, arvi])
                # Carbon
                labels.append([float(row[12])])

            line_count += 1

    data = DataFrame(data, columns = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    column_names = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category_Non woodland', 'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved', 'Species_Conifer', 'Species_Felled', 'Species_Grassland', 'Species_Young trees', 'Species_Other vegetation', 'NDVI', 'SAVI', 'EVI', 'ARVI']
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)

    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')

    return data, labels

def import_large_region(file):
    data = []
    labels = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1:
                #fid,CATEGORY,IFT_IOA,species1,sentinel1,sentinel2,landsat1,landsat21,landsat31,landsat41,landsat51,landsat61,landsat71,carbon
                # 0 fid 1 CATEGORY 2 species 3 interp_species1 4 sentinel1 5 sentinel2
                # 6 landsat1  7 landsat2 8 landsat3 9 landsat4
                # 10 landsat5 11 landsat6 12 landsat7
                # 13 carbon

                band2 = float(row[7])
                band4 = float(row[9])
                band5 = float(row[10])

                ndvi = (band5 - band4) / (band5 + band4)

                savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

                data.append([float(row[4]), float(row[5]), float(row[7]), float(row[8]), float(row[9]),  float(row[10]), float(row[11]), float(row[12]), row[1],  row[2], ndvi, savi, evi, arvi])
                # Carbon
                labels.append([float(row[13])])

            line_count += 1

    data = DataFrame(data, columns = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    column_names = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category_Non woodland', 'Category_Woodland', 'Species_Assumed woodland', 'Species_Broadleaved', 'Species_Conifer', 'Species_Felled', 'Species_Grassland', 'Species_Young trees', 'NDVI', 'SAVI', 'EVI', 'ARVI']
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)
    print(data)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')

    return data, labels

def import_region_grid(region,tag):
    data = []
    labels = []
    file = 'data/'+region+'_region/'+region+tag+'_grid_all.csv'
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1 and row[11] != '':
                # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
                # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
                # 8 Landsat5 9 Landsat6 10 Landsat7 11Carbon1


                band2 = float(row[5])
                band4 = float(row[7])
                band5 = float(row[8])

                ndvi = (band5 - band4) / (band5 + band4)

                savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

                data.append([float(row[2]), float(row[3]), float(row[5]), float(row[6]), float(row[7]), float(row[8]),  float(row[9]), float(row[10]),  row[0],  row[1], ndvi, savi, evi, arvi])
                # Carbon
                labels.append([float(row[11])])

            line_count += 1

    data = DataFrame(data, columns = ['Sentinel B1', 'Sentinel B2', 'Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    if region == 'train':
        column_names = TRAIN_COLUMNS
    else:
        column_names = OTHER_COLUMNS
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)

    #data.to_csv('data/small_region/grid_output.csv')
    #print(data.head(2))
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

    #print(data)
    #print(labels)
    #return data

def import_sentinel_grid(region,tag):
    print('in sentinel')
    data = []
    labels = []
    file = 'data/'+region+'_region/'+region+tag+'_grid_all.csv'
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1 and row[11] != '':
                # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
                # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
                # 8 Landsat5 9 Landsat6 10 Landsat7 11Carbon1
                data.append([float(row[2]), float(row[3]),  row[0],  row[1]])
                # Carbon
                labels.append([float(row[11])])

            line_count += 1

    data = DataFrame(data, columns = ['Sentinel B1', 'Sentinel B2', 'Category', 'Species'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    if region == 'train':
        column_names = TRAIN_COLUMNS_SENTINEL
    else:
        column_names = OTHER_COLUMNS_SENTINEL
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)

    #data.to_csv('data/small_region/grid_output.csv')
    #print(data.head(2))
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

    #print(data)
    #print(labels)
    #return data

def import_landsat_grid(region,tag):
    data = []
    labels = []
    file = 'data/'+region+'_region/'+region+tag+'_grid_all.csv'
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1 and row[11] != '':
                # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
                # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
                # 8 Landsat5 9 Landsat6 10 Landsat7 11Carbon1

                band2 = float(row[5])
                band4 = float(row[7])
                band5 = float(row[8])

                ndvi = (band5 - band4) / (band5 + band4)

                savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)

                data.append([float(row[5]), float(row[6]), float(row[7]), float(row[8]),  float(row[9]), float(row[10]),  row[0],  row[1], ndvi, savi, evi, arvi])
                # Carbon
                labels.append([float(row[11])])

            line_count += 1

    data = DataFrame(data, columns = ['Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
    labels = DataFrame(labels, columns = ['Carbon'])
    data = pd.get_dummies(data)
    if region == 'train':
        column_names = TRAIN_COLUMNS_LANDSAT
    else:
        column_names = OTHER_COLUMNS_LANDSAT
    data = data.drop(['Category_', 'Species_'], axis=1)
    data =  data.reindex(columns=column_names)

    #data.to_csv('data/small_region/grid_output.csv')
    #print(data.head(2))
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    return data, labels

    #print(data)
    #print(labels)
    #return data

def one_hot_to_csv(file):
    data = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1:

                # 0 fid, 1 CATEGORY, 2 IFT_IOA,
                # 3 COUNTRY, 4 Shape_Leng, 5 Shape__Are,
                # 6 x_coord, 7 y_coord


                data.append([row[1], row[2], float(row[4]), float(row[5])])

            line_count += 1

    data = DataFrame(data, columns = ['Category', 'Species', 'x_coord', 'y_coord'])
    data = pd.get_dummies(data)
    print(data.columns)
    print(data)
    # column_names = ['Category', 'Species', 'x_coord', 'y_coord']
    # data = data.drop(['Category_', 'Species_'], axis=1)
    # data =  data.reindex(columns=column_names)

    #data.to_csv('data/small_region/grid_output.csv')
    #print(data.head(2))
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')
    data.to_csv('data/total_region_onehot.csv')
    #for entry in forest_list:
        #print(f'{entry} appears {data['Vegetation'].count(entry)} times')

    #print(data)
    #print(labels)
    #return data

def main():
    #data, labels = import_region('data/medium_region/medium_grid_all.csv')
    #one_hot_to_csv('data/total_region_inven.csv')
    data, labels = import_train_region('data/train_region/train_grid_all.csv')

    # print(data['Species'].value_counts()/len(data))
    # print(data['Species'])

    # data['Species'] = data['Species'].map(species_dict)
    # data = data.astype({'Species': 'int32'})
    # data.to_csv('data/inventory_encoded.csv')
    # carbon = import_carbon_raster()
    # merged = import_merged_raster()
    # glob_data, glob_labels = import_grid('fromGlobal')
    # region_data, region_labels = import_grid('fromRegion')

if __name__ == "__main__":
    main()
