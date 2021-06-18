# import packages
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

import numpy as np
from random import random

import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats

from mpl_toolkits.axes_grid1 import make_axes_locatable

#remove later
from sklearn.model_selection import train_test_split
import pickle
import statistics

from rasterio.plot import show

from sklearn import svm
from model_params import get_extended_columns, get_columns

def plot_results(predictions, errors, y_test, filename):
    # Plot the actual values
    results = plt.figure(1)
    #plt.figure(figsize=(2,3))
    #plt.gcf().subplots_adjust(bottom=0.15, right=0.15, left=0.10)

    plt.plot(y_test, 'b-', label = 'actual', linewidth=0.5)

    # Plot the predicted values
    plt.plot(predictions, 'ro', label = 'prediction', linewidth=0.5)
    # Plot the errors
    plt.plot(errors, 'g-', label = 'errors', linewidth=0.5)
    plt.xticks(rotation = '60')
    plt.legend()
    # Graph labels
    plt.ylabel('Carbon Predicted')
    plt.title('Actual and Predicted Values')

    plt.savefig(filename + '_results.png')

    plt.close()

def plot_scatter(predictions, y_test, settings):
    filename = settings['filename']
    cap = ''
    if  settings['cap']:
        cap = ' (Capped)'
    title = settings['model'].capitalize() +' - '+ settings['dataset'].capitalize() + ' Carbon Scatter Plot' + cap
    # Plot the actual values
    scatter = plt.figure(random())
    plt.scatter(y_test, predictions, s=1)
    plt.xlabel('Actual AGB (Mg C/ha)')
    plt.ylabel('Predicted AGB (Mg C/ha)')
    plt.title(title)
    m, b = np.polyfit(y_test, predictions, 1)
    plt.plot(y_test, m*(y_test) + b, 'r-')
    plt.plot(y_test, y_test, color = 'black', linewidth=0.5)
    plt.gcf().set_size_inches(5, 5)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename + '_scatter.png')
    plt.close()

def draw_map(settings):
    model = settings['model']
    region = settings['region']
    filename = settings['filename']
    carbon = 'data/'+ region + '_region/raster/'+ region + '_carbon.tif'
    output_image = filename +'.tif'
    filename = filename+'_errorMap.png'

    output = rasterio.open(output_image)
    prediction = output.read()

    carbon = rasterio.open(carbon)
    carbon_label = carbon.read()

    print(carbon_label.shape)
    print(prediction.shape)

    difference = np.empty(carbon_label.shape, dtype = rasterio.float32)
    difference = carbon_label - prediction

    carbon_map = plt.figure(3)
    gridspec = {'width_ratios': [1, 1, 1, 0.1]}
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, nrows=1, figsize=(10, 3), gridspec_kw=gridspec)

    cap = ''
    if  settings['cap']:
        cap = ' (Capped)'
    title = settings['model'].capitalize() +' - '+ settings['dataset'].capitalize() + ' Carbon Map' + cap

    plt.suptitle(title, size=15)

    pred_plot = ax1.imshow(output.read(1), cmap = 'Greens')
    ax1.set_title('Prediction', size=10)

    plt.rcParams["axes.grid"] = False
    error_plot = ax3.imshow(difference[0], cmap='Greens')
    ax3.set_title('Error', size=10)
    ax3.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(True)
    carbon_plot = ax2.imshow(carbon.read(1), cmap='Greens')
    # asp = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
    # asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    # ax3.set_aspect(asp)
    ax2.set_title('Carbon Truth', size=10)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(True)
    plt.rcParams["axes.grid"] = False
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    plt.rcParams["axes.grid"] = False
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    divider = make_axes_locatable(ax4)
    cax = ax4
    cbar = plt.colorbar(carbon_plot, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Amount of carbon (Mg C/ha)', rotation=270, size=10)
    ax4.grid(True)
    carbon_map.patch.set_facecolor('white')
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()

def print_tree(tree, filename, features):
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = features, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('output/tree.dot')
    # Write graph to a png file
    graph.write_png('output/small_region/tree_' + filename + '.png')

def histogram(data, n_bins, x_label, y_label, title, cap=''):
    _, ax = plt.subplots()
    ax.hist(data, bins = n_bins, cumulative = False, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')
    plt.savefig('output/carbon_hist'+cap+'.png')

def var_importance(model, settings, features):
    model_name = settings['model']
    dataset = settings['dataset']
    file_dir = settings['filename']
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

def plot_topVar(model, settings, top):
    if settings['inven']:
        column_names = get_extended_columns(settings)
    else:
        column_names = get_columns(settings)

    top_feat = pd.Series(model.feature_importances_*100, index=column_names).nlargest(top)
    #print(top_feat)
    features = top_feat.plot(kind='barh')
    fig = features.get_figure()
    title = settings['model'].capitalize() + ' - ' + settings['dataset'].capitalize() + ' Top ' + str(top) + ' Variables'
    right_side = features.spines["right"]
    right_side.set_visible(False)
    top_side = features.spines["top"]
    top_side.set_visible(False)
    plt.ylabel('Most important features')
    plt.xlabel('Feature Importance (%)')


    for i, v in enumerate(top_feat.values.ravel()):
        features.text(v, i, str(round(v, 2))+'%', color='black', fontsize=7, ha='left', va='center')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title(title)

    fig.savefig(settings['filename']+'_topVar.png')

def main():
    settings = {
        'dataset': 'combined',
        'region': 'quick',
        'model': 'random_forest',
        'inven': True,
        'filename': 'tuned_noFEng',
        'cap': False,
        'log': False,
        'dim': '',
        'log_label': False,
    }
    settings['filename'] = settings['filename'] + '_' + settings['dataset']
    data, labels = import_grid(settings)
    upper_lim = np.quantile(labels, .95)
    print(f'Upper limit {upper_lim}')
    labels_95 = labels
    labels_97 = labels
    labels_95[labels_95 > upper_lim] = upper_lim
    labels_97[labels_97 > upper_lim] = upper_lim
    histogram(labels_95, 'Amount of Carbon Mg C/ha', 'Number of Instances', 'Label Carbon Histogram (Capped at 95%)', '95')
    histogram(labels_97, 'Amount of Carbon Mg C/ha', 'Number of Instances', 'Label Carbon Histogram (Capped at 97%)', '97')
    histogram(labels, 'Amount of Carbon Mg C/ha', 'Number of Instances', 'Label Carbon Histogram')

    # #labels = labels.values.ravel()
    # file_dir = 'output/'+ settings['region']+'_region/' + settings['model'] +'/' + settings['filename']
    # settings['filename'] = file_dir
    # plot_topVar(settings, 15)



if __name__ == "__main__":
    main()
