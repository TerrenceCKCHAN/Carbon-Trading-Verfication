import numpy as np

from cross_val import run, make_forest
from import_data import import_data, import_satellite_data, import_tree_carbon_data, import_satellite_raster, import_carbon_raster, import_grid

def main():

    rf = make_forest()
    satellite_input = import_satellite_raster()
    # print(satellite_input.shape)
    # print(satellite_input)
    predictions = rf.predict(satellite_input)

    print(predictions)

if __name__ == "__main__":
    main()
