# Carbon-Trading-Verification
This repository contains code used to the MSc Advanced Computing Thesis: Joint Study of Above Ground Biomass and Soil Organic Carbon for Total Carbon Estimation in Scotland. 

## Code Structure
Inside `scotland_carbon/src` contains our implementation:
- `train.py`: Trains the model, and output to file location specified by the user. Parameters include `model`, `target variable`, `machine learning technique`, `isLog`, `output path`.
- `feature_importance.py`: Generates the feature importance graph for specified models. Takes in `model`, `model path`, `output path`.
- `carbon_maps.py`: Provides two methods to generate carbon maps used in the report. The `plot_graph` function plots the prediction and error plots for the specified model. The `plot_single_graph` function plots for the carbon maps for total carbon estimation, total carbon ground truth and the total carbon error.
- `grid_search.py`: Contains the code for performing grid search on our models, we can tune hyperparameters for any model and ML technique. Parameters are `model`, `target variable`, `ml technique` and `isLog` 
- `utils.py` contains utility functions and model definitions 


## File Structure
- `archive` contains previous work from Carla `Estimating-AGB` and Anish `Estimating-SOC`.
- `scotland_carbon` contains our implementation

## Trained Models
The trained models and the results are available on AWS (https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/project_models.zip). Below is a brief description of the features used in different models
- `VH_1` `VV_1` corresponds to the Sentinel 1 data obtained
- The prefix `BAND_` indicate data source from Sentinel 2
- The prefix `L_` indicate data source from LandSat 8 satellite
- `DEM_CS` `DEM_LSF` `DEM_TWI` `DEM_ELEV` stands for digital elevation derivatives obtained from the digital elevation map, they corresponds to catchment slope, length slope factor , tropical wetness index and digital elevation respectively.
- `NDVI` `EVI` `SATVI` are vegetation indices calculated from Sentinel 2
- `CATEGORY` corresponds to data from the forest inventory data

## Remote Sensing Data
All remote sensing data including Sentinel 1, Sentinel 2, LandSat 8, DEM, Inventory Data is available on AWS at:
- https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/satellite_data_part1.zip
- https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/satellite_data_part2.zip
- https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/satellite_data_part3-1.zip
- https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/satellite_data_part3-2.zip
- https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/satellite_data_part4.zip

## Virtual Environment Location
https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-terrence/venv.zip

## Running the code
Each file can be run separately by calling the corresponding functions. 
- The csv file location is `Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv`.
- The Evaluation tif file for Model G is `Carbon-Trading-Verification\scotland_carbon\data\MODEL_G_EVAL.tif`
- The Evaluation tif file for Model H is `Carbon-Trading-Verification\scotland_carbon\data\MODEL_H_EVAL.tif`
- Note that for the `isLog` parameter, we use log for SOC estimation and no log for AGB estimation, this is based on prior experimentations that taking log improves model training and performance for SOC estimation but not the case for AGB estimation. 
