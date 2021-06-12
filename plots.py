# import packages
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

import numpy as np

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats

from mpl_toolkits.axes_grid1 import make_axes_locatable

from rasterio.plot import show

from sklearn import svm
from import_data import import_satellite_data, import_tree_carbon_data, import_region_grid

def plot_results(predictions, errors, y_test, filename):
    # Plot the actual values
    results = plt.figure(1)
    plt.plot(y_test, 'b-', label = 'actual')
    # Plot the predicted values
    plt.plot(predictions, 'ro', label = 'prediction')
    # Plot the errors
    plt.plot(errors, 'g-', label = 'errors')
    plt.xticks(rotation = '60');
    plt.legend()
    # Graph labels
    plt.ylabel('Carbon Predicted'); plt.title('Actual and Predicted Values');
    plt.savefig(filename + '_results.png')

def plot_scatter(predictions, y_test, title, filename):
    # Plot the actual values
    scatter = plt.figure(2)
    plt.scatter(y_test, predictions, s=1)
    plt.xlabel('Actual Carbon Mg C/ha')
    plt.ylabel('Predicted Carbon Mg C/ha')
    plt.title('Actual and Predicted Carbon Scatter Plot')
    m, b = np.polyfit(y_test, predictions, 1)
    plt.plot(y_test, m*(y_test) + b, 'r-')

    plt.savefig(filename + '_scatter.png')

def draw_map(model, region, filename):
    carbon = 'data/'+ region + '_region/raster/'+ region + '_carbon.tif'
    output_image = 'output/'+ region + '_region/raster/prediction_map_'+ model + '.tif'

    output = rasterio.open(output_image)
    prediction = output.read()

    carbon = rasterio.open(carbon)
    carbon_label = carbon.read()

    print(carbon_label.shape)
    print(prediction.shape)

    # carbon_label = np.repeat(carbon_label, 6,  axis=1)
    # carbon_label = np.repeat(carbon_label, 6,  axis=2)
    # print(carbon_label.shape)
    # carbon_label = np.delete(carbon_label, -1, axis=1)
    # if region == 'small':
    #     carbon_label = np.delete(carbon_label, [-2,-1], axis=2)
    # elif region == 'medium':
    #     carbon_label = np.delete(carbon_label, -1, axis=2)

    difference = np.empty(carbon_label.shape, dtype = rasterio.float32)
    difference = carbon_label - prediction

    carbon_map = plt.figure(3)
    gridspec = {'width_ratios': [1, 1, 1, 0.1]}
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(10, 3), gridspec_kw=gridspec)

    plt.suptitle('Carbon Map Results Comparison (using '+ model + ' model)', size=15)

    pred_plot = ax1.imshow(output.read(1), cmap = 'viridis')
    ax1.set_title('Prediction', size=10)
    error_plot = ax2.imshow(difference[0], cmap='viridis')
    ax2.set_title('Difference', size=10)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(True)
    carbon_plot = ax3.imshow(carbon.read(1), cmap='viridis')
    # asp = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
    # asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    # ax3.set_aspect(asp)
    ax3.set_title('Carbon Truth', size=10)
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    divider = make_axes_locatable(ax4)
    cax = ax4
    cbar = plt.colorbar(carbon_plot, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Amount of carbon (Mg C/ha)', rotation=270, size=7)

    plt.rcParams["axes.grid"] = False
    plt.savefig(filename, bbox_inches = 'tight')

def print_tree(tree, filename, features):
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = features, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('output/tree.dot')
    # Write graph to a png file
    graph.write_png('output/small_region/tree_' + filename + '.png')

def plot_var_rank(model, model_name, features, filename):
    # Get numerical feature importances
    if model_name == 'linear':
        importances = model.coef_
    else:
        importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # Plot importances
    # Set the style
    plt.tight_layout()
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, features, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

    plt.savefig(filename + '.png')
    return importances

def histogram(data, n_bins, x_label = "", y_label = "", title = "", cap="" ,cumulative=False):
    _, ax = plt.subplots()
    ax.hist(data, bins = n_bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.savefig('output/carbon_hist'+cap+'.png')

def main():

    #draw_map('random_forest', 'quick', 'output/quick_region/random_forest/nothing300_lablog')
     data, labels = import_region_grid('quick', '300')
     labels = labels.values.ravel()
     upper_lim = np.quantile(labels, .95)
     print(f'Upper limit {upper_lim}')
     labels_cap = labels
     labels_cap[labels_cap > upper_lim] = upper_lim
     histogram(labels, 15, x_label='Amount of Carbon (Mg C/ha)', y_label='Number of Instances', title='Labeled Carbon Histogram (Original Data)')
     histogram(labels_cap, 15, x_label='Amount of Carbon (Mg C/ha)', y_label='Number of Instances', title='Labeled Carbon Histogram (Capped Data at 95%)',cap='_cap')

    #
    #
    # _ = plt.hist(labels, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Carbon label data distribution")
    # plt.show()
    #
    # winsorize(labels, limits=[0, 0.3])
    #
    # _ = plt.hist(labels, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Carbon label data distribution (winsorized)")
    # plt.show()

if __name__ == "__main__":
    main()
