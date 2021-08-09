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

model_path = {
    'soc_brt': "../../models/joint_models/soc/brt_model.joblib.pkl"
}

input_path = {
    'soc_features': fr'C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\S1A_S2AL2A_INDICES_DEM_EVAL.tif'
}

output_path = {
    'soc_brt_features': '../../report_output/exp1-joint/carbon_maps/soc/soc_brt_features.png'
}

def evaluate(model_path, input_path, isLog):

    model = joblib.load(model_path)
    image = rasterio.open(input_path)

    num_bands = image.count
    all_data = []

    print("Converting Raster to Array...")
    for i in tqdm(range(num_bands)):
        data = image.read(i+1)
        data = pd.DataFrame(data).fillna(0).to_numpy()
        all_data.append(data)

    all_data = np.dstack(all_data)
    result_data = []

    for t in tqdm(all_data):
        z = model.predict(t)
        result_data.append(z)
        z[z!=z] = 0

    result_data = np.exp(np.stack(result_data)) if isLog else np.stack(result_data)

    return result_data




def plot_graph(result_data, err_data, title, map_style, out_file_name):

    fontprops = fm.FontProperties(size=12)
    x_coord_l, x_coord_r = -3.7037717, -2.7696528
    y_coord_t, y_coord_b = 55.7541937, 55.4075244
    x_shape = result_data.shape[1]
    y_shape = result_data.shape[0]

    def x_format_func(value, tick_number):
        out = (x_coord_l + (value / x_shape * (x_coord_r - x_coord_l)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    def y_format_func(value, tick_number):
        out = (y_coord_t - (value / y_shape * (y_coord_t - y_coord_b)))
        return "{v:.2f}\N{DEGREE SIGN}".format(v=out)

    ###################################################################################
    gridspec = {'width_ratios': [1, 1, 0.1]}
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(9, 4), gridspec_kw=gridspec)
    fig.suptitle(title)
    
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(x_format_func))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(y_format_func))

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

    ax1.add_artist(scalebar1)
    ax2.add_artist(scalebar2)
    ax1.set_title('Prediction Map', size=10)
    ax2.set_title('Error Map', size=10)
    

    pred_plot = ax1.imshow(result_data, cmap = map_style)
    ax2.imshow(err_data, cmap = map_style)
    
    ax2.get_yaxis().set_visible(False)
    
    fig.text(0.5, 0.12, 'Longitude', ha='center')
    fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical')
    
    cbar = plt.colorbar(pred_plot, cax=ax3)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Amount of carbon (Mg C/ha)', rotation=270, size=8)
    
    plt.savefig(out_file_name)
    
# soc_truth = rasterio.open(fr'C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\Train_SG_15-30_27700_clipped.tif').read(1)
soc_truth = rasterio.open(fr'C:\Users\admin\OneDrive\Computing\Yr5 Advanced Computing\MAC Project\Carbon-Trading-Verification\scotland_carbon\data\Train_SOCS_0-30_27700_clipped.tif').read(1)
soc_result = evaluate(model_path['soc_brt'], input_path['soc_features'], True)
# pred, error, title, output directory
plot_graph(np.clip(soc_result,65,130), np.abs(soc_truth - soc_result[:131, :193]), 'BRT SOC Carbon Map', 'Greens', output_path['soc_brt_features'])