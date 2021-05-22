#import sklearn
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification

import numpy as np

from sklearn import svm
from import_data import import_satellite_data, import_tree_carbon_data

def main():

    data, labels = import_satellite_data()

    print(data.describe())

    #species, carbon = import_tree_carbon_data()

    # num_rows = data.shape[0]
    # random_indices = np.random.choice(num_rows, size=10000, replace=False)
    # #random_rows = an_array[random_indices, :]
    #
    # #print(f'Row indices chosen: {random_indices}')
    # piece_data= data[random_indices, :]
    # piece_labels = labels[random_indices, :]


    X_train, X_test, y_train, y_test = train_test_split(data, labels.ravel(), test_size=0.25, random_state=0)

    clf = RandomForestRegressor(n_estimators=500, max_depth=3)
    #print(f'labels non ravel {labels}')
    #print(f'labels ravel {labels.values.ravel()}')
    #print(f'data {data}')
    scores = cross_val_score(clf, data, labels.values.ravel(),scoring='neg_root_mean_squared_error')
    print(f'scores {scores}')
    #clf.fit(X_train, y_train)
    #print(clf.score(X_test, y_test))

    #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    #print(clf.score(X_test, y_test))

    #clf = svm.SVC(kernel='linear', C=1, random_state=42)
    #scores = cross_val_score(clf, data, labels.ravel())

    # Split the data into training, validation and test set


if __name__ == "__main__":
    main()
