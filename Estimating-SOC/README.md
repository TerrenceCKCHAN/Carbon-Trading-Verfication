# individual_project

This repository contains code used for the final year project, "Spatial Finance for Sustainable Development: Soil Carbon Measurement". This inclused code for training and evaluating multiple models, and generating raster files of carbon maps.

## Structure
<ul>
<li>
    data: contains csv data files used to train models. Each csv represents a list of pixel points from an image with spectral, radar and topological band values. These csv files were generated using QGIS and raster data available on [AWS](https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-anish/qgis.rar)
    <ul>
    <li>
        S1AIW includes Sentinel-1A bands from multiple numbered images
    </li>
    <li>
        S2AL2A includes Sentinel-2A L2A bands
    </li>
    <li>
        NDVI, EVI, SATVI are vegetation indices derived from Sentinel-2 bands
    </li>
    <li>
        DEM includes Digital Elevation Map data and derived information such as catchment slope, slope, length slope factor, topographic wetness index
    </li>
    <li>
        LUCASTIN includes Triangulated Irregular Network Interpolated soil carbon label data from the LUCAS 2009 set
    </li>
    <li>
        roi_points_0.02 denotes that these are region of interest data points with a spacing of 0.02 degrees
    </li>
    <li>
        LUCAS2009_zhou2020_points denotes the original LUCAS data points were used from the [Zhou et al. 2020 paper](https://www.sciencedirect.com/science/article/pii/S0048969720317575?via%3Dihub)
    </li>
    <li>
        SoilGrids2_0 denotes the use of SoilGrids2.0 label data instead of interpolated LUCAS data
    </li>
    </ul>
</li>
<li>
    models: Contains the models generated after training on csv files. The carbon maps of the best models can be viewed in QGIS using the data found in [AWS](https://bci-satellite-carbon.s3.eu-west-2.amazonaws.com/imperial-anish/qgis.rar). Also contains txt files of model performance on independent and lucas test sets for reference. Most models were trained on spacing of 0.04 and LUCAS data unless specified otherwise.
</li>
<li>
    out: Contains grid search output files and other useful graphs and maps generated by model training and evaluation. Most of these were not used in the final report as final graphs and maps were better rendered in QGIS.
</li>
<li>
    src: Contains all the short python scripts for this project. Organised by their model type, all scripts are run by typing "python script.py" with no additional parameters. Any paths and parameters should be changed within the scripts as relevant. Most models use scikit-learn, nns use Pytorch and utils use a few different statistical and graph plotting libraries (statsmodel, sns, matplotlib, scipy). Scikit-learn models run fairly quickly and can be trained within minutes using multiple CPU cores, Pytorch models take slightly longer when run on GPU. utils/getwcs.py is used to obtain WCS files, using owslib, for the SoilGrids label data as this feature did not work in QGIS
</li>
</ul>