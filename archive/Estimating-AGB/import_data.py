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

from model_params import get_columns, get_extended_columns


def import_grid(settings):
    data = []
    labels = []
    region = settings['region']
    dataset = settings['dataset']
    file = 'data/'+region+'_region/'+region+'_grid_all.csv'
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = -1
        forest_list = []
        for row in csv_reader:
            # Ensure it is not the first row and there are satellite readings
            if line_count > -1 and row[11] != '':
                input_row = []
                # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
                # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
                # 8 Landsat5 9 Landsat6 10 Landsat7 11 Carbon1
                if dataset != 'landsat':
                    input_row.extend([float(row[2]), float(row[3])])

                if dataset != 'sentinel':
                    band2 = float(row[5])
                    band4 = float(row[7])
                    band5 = float(row[8])

                    ndvi = (band5 - band4) / (band5 + band4)

                    savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5

                    evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))

                    arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)
                    input_row.extend([float(row[5]), float(row[6]), float(row[7]), float(row[8]),  float(row[9]), float(row[10])])
                if settings['inven']:
                    input_row.extend([row[0],  row[1]])

                if dataset != 'sentinel':
                    input_row.extend([ndvi, savi, evi, arvi])
                data.append(input_row)
                # Carbon
                labels.append([float(row[11])])

            line_count += 1

    column_names = get_columns(settings)
    data = DataFrame(data, columns = column_names)
    labels = DataFrame(labels, columns = ['Carbon'])
    # One-hot encoding
    data = pd.get_dummies(data)
    if settings['inven']:
        column_names = get_extended_columns(settings)
        data = data.drop(['Category_', 'Species_'], axis=1)
        data = data.reindex(columns=column_names)

    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')

    return data, labels

# def import_sentinel_grid(region,tag):
#     print('in sentinel')
#     data = []
#     labels = []
#     file = 'data/'+region+'_region/'+region+tag+'_grid_all.csv'
#     with open(file) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = -1
#         forest_list = []
#         for row in csv_reader:
#             # Ensure it is not the first row and there are satellite readings
#             if line_count > -1 and row[11] != '':
#                 # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
#                 # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
#                 # 8 Landsat5 9 Landsat6 10 Landsat7 11 Carbon1
#                 data.append([float(row[2]), float(row[3]),  row[0],  row[1]])
#                 # Carbon
#                 labels.append([float(row[11])])
#
#             line_count += 1
#
#     if dataset == 'combined':
#         columns = COMBINED_COLUMNS
#     elif dataset == 'landsat':
#         columns == LANDSAT_COLUMNS
#     else:
#         COLUMNS = SENTINEL_COLUMNS
#
#     data = DataFrame(data, columns = ['Sentinel VH', 'Sentinel VV', 'Landsat 2', 'Landsat 3', 'Landsat 4', 'Landsat 5', 'Landsat 6', 'Landsat 7', 'Category', 'Species'])
#     labels = DataFrame(labels, columns = ['Carbon'])
#     data = pd.get_dummies(data)
#     if region == 'train':
#         column_names = TRAIN_COLUMNS_SENTINEL
#     else:
#         column_names = OTHER_COLUMNS_SENTINEL
#     data = data.drop(['Category_', 'Species_'], axis=1)
#     data =  data.reindex(columns=column_names)
#
#     #data.to_csv('data/small_region/grid_output.csv')
#     #print(data.head(2))
#     #data = np.asarray(data)
#     #labels = np.asarray(labels)
#     print(f'Processed {line_count} lines.')
#     print(f'Entries in data {len(data)}.')
#     #print(f'Forest list {forest_list}.')
#
#     #for entry in forest_list:
#         #print(f'{entry} appears {data['Vegetation'].count(entry)} times')
#
#     return data, labels
#
#     #print(data)
#     #print(labels)
#     #return data
#
# def import_landsat_grid(region,tag):
#     data = []
#     labels = []
#     file = 'data/'+region+'_region/'+region+tag+'_grid_all.csv'
#     with open(file) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         line_count = -1
#         forest_list = []
#         for row in csv_reader:
#             # Ensure it is not the first row and there are satellite readings
#             if line_count > -1 and row[11] != '':
#                 # 0 CATEGORY  1 IFT_IOA 2 SAMPLE_1 (sentine1) 3 SAMPLE_2
#                 # 4 Landsat1 5 Landsat2 6 Landsat3 7 Landsat4
#                 # 8 Landsat5 9 Landsat6 10 Landsat7 11Carbon1
#
#                 band2 = float(row[5])
#                 band4 = float(row[7])
#                 band5 = float(row[8])
#
#                 ndvi = (band5 - band4) / (band5 + band4)
#
#                 savi = ((band5 - band4) / (band5 + band4 + 0.5)) * 1.5
#
#                 evi = 2.5 * ((band5 - band4) / (band5 + 6 * band4 - 7.5 * band2 +1))
#
#                 arvi = (band5 - 2*band4 - band2) / (band5 + 2*band4 - band2)
#
#                 data.append([float(row[5]), float(row[6]), float(row[7]), float(row[8]),  float(row[9]), float(row[10]),  row[0],  row[1], ndvi, savi, evi, arvi])
#                 # Carbon
#                 labels.append([float(row[11])])
#
#             line_count += 1
#
#     data = DataFrame(data, columns = ['Landsat2', 'Landsat3', 'Landsat4', 'Landsat5', 'Landsat6', 'Landsat7', 'Category', 'Species', 'NDVI', 'SAVI', 'EVI', 'ARVI'])
#     labels = DataFrame(labels, columns = ['Carbon'])
#     data = pd.get_dummies(data)
#     if region == 'train':
#         column_names = TRAIN_COLUMNS_LANDSAT
#     else:
#         column_names = OTHER_COLUMNS_LANDSAT
#     data = data.drop(['Category_', 'Species_'], axis=1)
#     data =  data.reindex(columns=column_names)
#
#     #data.to_csv('data/small_region/grid_output.csv')
#     #print(data.head(2))
#     #data = np.asarray(data)
#     #labels = np.asarray(labels)
#     print(f'Processed {line_count} lines.')
#     print(f'Entries in data {len(data)}.')
#     #print(f'Forest list {forest_list}.')
#
#     #for entry in forest_list:
#         #print(f'{entry} appears {data['Vegetation'].count(entry)} times')
#
#     return data, labels
#
#     #print(data)
#     #print(labels)
#     #return data

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
    print("Import data main")

if __name__ == "__main__":
    main()
