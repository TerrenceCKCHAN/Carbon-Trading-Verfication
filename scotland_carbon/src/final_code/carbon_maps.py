import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.crs import CRS
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Please specify model location
model_paths = {
    'model_G': "../../models/soc/rf_model_G.joblib.pkl",
    'model_H': "../../models/agb/rf_model_H.joblib.pkl",
}

# Please specify input location
input_paths = {
    'model_G': fr'Carbon-Trading-Verification\scotland_carbon\data\MODEL_G_EVAL.tif',
    'model_H': fr'Carbon-Trading-Verification\scotland_carbon\data\MODEL_H_EVAL.tif'
}

# Evaluate raster files on machine learning models and return the resulting estimation
def evaluate(model_path, input_path, isLog):

    # Load model and input images
    model = joblib.load(model_path)
    image = rasterio.open(input_path)

    # Specify number of bands
    num_bands = image.count
    # Intialise data array
    all_data = []

    # Convert raster to array
    print("Converting Raster to Array ...")
    for i in tqdm(range(num_bands)):
        data = image.read(i+1)
        data = pd.DataFrame(data).fillna(0).to_numpy()
        all_data.append(data)

    all_data = np.dstack(all_data)

    # Intialise result data array
    result_data = []

    # Populate result data array with predictions
    for t in tqdm(all_data):
        z = model.predict(t)
        result_data.append(z)
        z[z!=z] = 0

    # Get exponent of result data if original model is logged
    result_data = np.exp(np.stack(result_data)) if isLog else np.stack(result_data)
    # Return carbon estimation
    return result_data


# Function to plot prediction data and error data
def plot_graph(result_data, err_data, title, map_style, out_file_name):

    # Set font
    fontprops = fm.FontProperties(size=12)
    # Coordinates corresponding to our study area
    x_coord_l, x_coord_r = -3.7037717, -2.7696528
    y_coord_t, y_coord_b = 55.7541937, 55.4075244
    # Store data shape
    x_shape = result_data.shape[1]
    y_shape = result_data.shape[0]

    # Format x coordinates
    def x_format_func(value, tick_number):
        out = (x_coord_l + (value / x_shape * (x_coord_r - x_coord_l)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    # Format y coordinates
    def y_format_func(value, tick_number):
        out = (y_coord_t - (value / y_shape * (y_coord_t - y_coord_b)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    ###################################################################################
    # Specify grid ratio
    gridspec = {'width_ratios': [1, 1, 0.1]}
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(9, 4), gridspec_kw=gridspec)
    
    # Set formatter for x and y axis
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(y_format_func))

    # Set scalebar for left and right subplot
    scalebar1 = AnchoredSizeBar(ax1.transData,
                               33.3, '10 km', 'lower right', 
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)
    
    scalebar2 = AnchoredSizeBar(ax2.transData,
                               33.3, '10 km', 'lower right', 
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)

    # Add scale bar to subplots
    ax1.add_artist(scalebar1)
    ax2.add_artist(scalebar2)
    # Set title to subplots
    ax1.set_title('Prediction Map', size=10)
    ax2.set_title('Error Map', size=10)
    
    # Set prediction plot and error plot with colour map
    pred_plot = ax1.imshow(result_data, cmap = map_style)
    err_plot = ax2.imshow(err_data, cmap = map_style)
    ax2.get_yaxis().set_visible(False)
    
    # Set figure labels - latitude and longitude
    fig.text(0.5, 0.12, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    
    # Set colour bar and its axis
    cbar = plt.colorbar(pred_plot, cax=ax3)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Amount of carbon (Mg C/ha)', rotation=270, size=8)
    
    # Save file output
    plt.savefig(out_file_name)


def plot_single_graph(result_data, map_style, out_file_name):
    
    # Set font
    fontprops = fm.FontProperties(size=12)
    # Coordinates corresponding to our study area    
    x_coord_l, x_coord_r = -3.7037717, -2.7696528
    y_coord_t, y_coord_b = 55.7541937, 55.4075244
    # Store data shape
    x_shape = result_data.shape[1]
    y_shape = result_data.shape[0]

    # Format x coordinates
    def x_format_func(value, tick_number):
        out = (x_coord_l + (value / x_shape * (x_coord_r - x_coord_l)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    # Format y coordinates
    def y_format_func(value, tick_number):
        out = (y_coord_t - (value / y_shape * (y_coord_t - y_coord_b)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    ###################################################################################
    # Set formatter for x and y axis
    ax = plt.axes()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_format_func))

    # Set scalebar for plot
    scalebar = AnchoredSizeBar(ax.transData,
                               33.3, '10 km', 'lower right', 
                               pad=0.1,
                               color='black',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)

    # Add scalebar to plot
    ax.add_artist(scalebar)

    # Set figure labels - latitude and longitude
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # Set figure colour bar and map style
    plt.imshow(result_data, cmap=map_style)
    plt.colorbar(label="Amount of Organic Carbon (Mg C/ha)")   
    # Save figure to destination
    plt.savefig(out_file_name)


# Get SOC Ground Truth
soc_truth = rasterio.open(fr'C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\Train_SOCS_0-30_27700_clipped.tif').read(1)
# Get SOC Prediction
soc_result = evaluate(model_paths['model_G'], input_paths['model_G'], True)
# Uncomment to plot soc prediction and error graphs
# plot_graph(np.clip(soc_result, 50,120), np.abs(soc_truth - soc_result[:131, :193]), 'Greens', '.')

# Get AGB Ground Truth
agb_truth = rasterio.open(fr'C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\agb_c1_27700_300m_clipped.tif').read(1)
# Get AGB Prediction
agb_result = evaluate(model_paths['model_H'], input_paths['model_H'], False)
# Uncomment to plot agb predicion and error graphs
# plot_graph(agb_result, np.abs(agb_truth - agb_result[:130, :192]), 'YlGn', '.')

# Uncomment to plot total carbon ground truth 
# plot_single_graph(
#     agb_truth+soc_truth[:130,:192],
#     'YlOrRd' ,
#     '../../report_output/exp2/carbon_maps/total_carbon_map_truth.png'
#     )

# Uncomment to plot total carbon error
# plot_single_graph(
#     np.abs((agb_result+soc_result)[:130,:192] - (agb_truth+soc_truth[:130,:192])),
#     'YlOrRd' ,
#     '../../report_output/exp1-joint/carbon_maps/total_carbon_map_err.png'
#     )

# Uncomment to plot total carbon prediction
# plot_single_graph(
#     agb_result+soc_result, 
#     'YlOrRd' ,
#     '../../report_output/exp1-joint/carbon_maps/total_carbon_map_pred.png'
#     )



