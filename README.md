# Carbon-Trading-Verfication
This repository contains code used to the MSc Advanced Computing Thesis: Joint Study of Above Ground Biomassand Soil Organic Carbon for Total Carbon Estimation in Scotland. 

## Code Structure
Inside `scotland_carbon/src` contains our implementation:
- `train.py` trains the model
- `feature_importance.py` generates the feature importance graph for specified models
- `carbon_maps.py` provides two methods to generate carbon maps used in the report
- `grid_search.py` contains the code for performing grid search on our models
- `utils.py` contains utility functions and model definitions 


## File Structure
- `archive` contains previous work from Carla and Anish.
- `scotland_carbon` contains our implementation

## Running the code
Each file can be run separately by calling the corresponding functions. 
- The csv file location is `Carbon-Trading-Verification\scotland_carbon\data\S1AIW_S2AL2A_DEM_IDX_SOCS_SG_L_INVEN_AGB_300m_processed.csv`.
- The Evaluation tif files for Model G is `Carbon-Trading-Verification\scotland_carbon\data\MODEL_G_EVAL.tif`
- The Evaluation tif files for Model H is `Carbon-Trading-Verification\scotland_carbon\data\MODEL_H_EVAL.tif`
