#from osgeo import gdal
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

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

# Classes with less than 5 instances
NOT_INCLUDE = ['Uncertain', 'Coppice', 'Power line', 'River']



def not_empty_entries(row, indeces):
    for index in indeces:
        if row[index] == None or row[index] == '':
            return False
    return True

def check_possible(line_count, row):
    return line_count > 0 and not_empty_entries(row, [12,13]) and row[4] not in NOT_INCLUDE

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
            if check_possible(line_count, row):
                forest_list, forest_id = find_dict_num(forest_list, row[3], row[4])
                # CATEGORY, IFT_IOA, Shape_Leng, Area_ha, Shape__Are, Shape__Len, x_coord, y_coord, satellite1, satellite2
                #print(f'appending {forest_id} for {row[3], row[4]} tuple')
                data.append([row[3], row[4], float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13])])
                # Carbon
                labels.append([float(row[14])])

            line_count += 1

    data = DataFrame(data, columns = ['Category', 'Species', 'Shape_Leng', 'Area_ha', 'Shape__Are', 'Shape__Len', 'x_coord', 'y_coord', 'satellite1', 'satellite2'])
    labels = DataFrame(labels, columns = ['Carbon'])
    #data = np.asarray(data)
    #labels = np.asarray(labels)
    print(f'Processed {line_count} lines.')
    print(f'Entries in data {len(data)}.')
    #print(f'Forest list {forest_list}.')

    # One-hot encode the data using pandas get_dummies
    data = pd.get_dummies(data)
    
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
