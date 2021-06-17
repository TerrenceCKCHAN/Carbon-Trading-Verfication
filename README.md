# Estimating AGB

The Estimating AGB repository contains the code used for the final year project: Estimating Aboveground Biomass Using Remote Sensors with Machine Learning Algorithms. This includes the code for the feature engineering, plots and carbon maps built for the different models explained in the report.

## Code Structure
The code is separated into several files:
- `main.py`:  contains the main code execution. Uses support functions from the rest of the files. It sets the execution settings (explained below) and can run a complete model train, test, carbon mapping and plotting by calling `run(settings)`
- `import.py`: used to correctly input and format the training grid. It controls the exact features are present, depending on what data source is used and performs one hot encoding on the inventory data, if present.
- `plots.py`: contains most, if not all, of the code for the generated plots in the report. These include: the scatter plots, histograms, feature importances and carbon maps. Other plotting functions are also. available (such as `plot_results` or `draw_tree`) that were not included because the high volume of data render them unuseful.
- `feature_engineering.py`: contains the implementation for the feature engineering methods implemented (outlier capping,  log transformation of the data and/or the labels and dimensionality reduction)
- `model_params.py`: contains information specific to each model, region and type of run. For instance, it stores the tuned optimised model parameters which can be accessed by calling `get_params(settings)`.  Moreover, when importing the grid or building the prediction raster file, different features are used as input. For example, using the `sentinel` dataset only does not include any of the Landsat 8 bands or derived vegetation indices. Therefore, careful attention is required to ensure the column names and length match with each input. This is taken care of with the `get_inven_abv` (used to access the inventory input raster files), `get_columns` (used prior to one-hot encoding) and `get_extended_columns` (used after one-hot encoding).
- `tuning.py`: contains the code used to prune the random forest and XGBoost models. The XGBoost tuning can be done in smaller groups by calling `tune_xgboost` (this was the method used for the report) or in one go (calling `hyperparameter_tuning`). Please note, doing a grid search over the 5 XGBoost parameters will take a lot of time.
## Model settings

```bash
settings = {
    'dataset': 'combined',
    'region': 'quick',
    'model': 'linear',
    'inven': True,
    'filename': 'example',
    'cap': 0,
    'log': False,
    'log_label': False,
    'dim': 'pca',   
}
```


Most of the analysis can be carried out by changing the parameter values of the settings function in the main file:

- The `dataset` variable determines what data sources to use. The options are: `landsat`, `sentinel` or `combination`.
- The `region` variable can be set to `quick` (the region used to train the models for the report) or `train`(the region used as unseen data to evaluate the outputted carbon map). The models can be trained on both regions as the grid-data is also available for the `test` region. This will, however take considerable time to run.
-  The `model`. entry controls the machine learning model trained. `linear`, `random_forest` and `xgboost` supported. Please note that the `linear` model will not produce a feature importance plot as all the variables were used to build the linear relationships for the estimates.
- The `inven` region can be toggled between `True` (include the inventory data in the input) or `False`. It was set to false to analyse the performance of the models without the National Forestry Inventory (NFI) data.
- `filename` (later updated to pass the whole file directory) is the variable set to identify the model's output files from the rest of the instances run.  All of the files generated in this run will contain the `filename` as a base.

The remaining settings parameters are used to test different feature engineering methods:
- `cap` can be a number between `0` and `1` (`0` being no cap, the default setting). A `cap` value of `0.05`, for example, caps the label data at 95%.
-  `log` when set to `True` performs a log transformation of the numerical features in the input
- Similarly, `log_label` determines whether or not to use the log transform of the labeled data. When this is set to `True`, the predictions made for the carbon map are converted back to the true scale to be comparable to the actual labeled data.
-`dim` can be either empty (no dimensionality reduction), `pca` or `pda`. The number of features remaining can be changed, its default being 7.
Please note that when the `dim` setting is not empty, each model uses a different set of input variables are therefore if there is a need to compute the carbon map, it is left as future work.

## File Organisation
The input and output files are split into `input` and `output` folders respectively. The input folder is further divided among the different regions. Inside each region there is the grid (training) file and a raster folder. As the name indicates, the raster folder contains the necessary raster files (for both satellites and all inventory species) to create the prediction raster. The abbreviations used for the inventory raster files are:
- _ag: Agriculture land (Species)
- _as: Assumed woodland (Species)
- _ba: Bare area (Species)
- _br: Broadleaved (Species)
- _co: Conifer (Species)
- _cop: Coppice (Species)
- _fa: Failed (Species)
- _fe: Felled (Species)
- _gr: Grassland (Species)
- _gro: Ground prep (Species)
- _lo: Low density (Species)
- _mib: Mixed mainly broadleaved (Species)
- _mic: Mixed mainly conifer (Species)
- _nonwood: Non Woodland (Category)
- _op: Open water (Species)
- _ot: Other vegetation (Species)
- _po: Power line (Species)
- _qu: Quarry (Species)
- _ri: River (Species)
- _ro: Road (Species)
- _sh: Shrub (Species)
- _un: Uncertain (Species)
- _ur: Urban (Species)
- _wib: Windblow (Species)
- _wif: Windfarm (Species)
- _wood: Woodland (Category)
- _yo: Young trees (Species)

The output files are organised by region first as well, and then subcategorised by model. The generated plots, raster files and carbon maps are stored in there.

Please note that before using the code the files must be downloaded from AWS.

## Generated plots
When `run` is called, the program builds, trains, tests and analyses the model specified by the `settings`. As part of the analysis, the program builds several plots and stores them in the correct subdirectory, named after `filename` and with the corresponding abbreviation:
- _scatter
- _errorMap
- _topVar
- .sav (saved model)
- .tif (prediction raster)

There is also the option to plot the histogram of the data distribution.


## Execution
As mentioned beforehand, the program can be executed from main file by calling `run` on different setting configurations. Please note the filename is modified by the code for each dataset (ie the code will reflect a  `dataset` change from `sentinel` to `landsat` in the outputted files (from xxx_sentinel_scatter to xxx_landsat_scatter).
