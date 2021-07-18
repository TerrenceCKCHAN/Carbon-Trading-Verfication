
def xgb_maxdepth_childweight(params, dtrain, dtest):

    params['eval_metric'] = "mae"
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(2,12,2)
        for min_child_weight in range(1,6)
    ]

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
                                 max_depth,
                                 min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth,min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    return best_params[0], best_params[1], min_mae
    # printed: Best params: 6, 2, MAE: 98.3690782

def eta_sub_tuning(params, dtrain, dtest):

    gridsearch_params = [
        (subsample, eta)
        for subsample in [i/10. for i in range(6,11)]
        for eta in [0.01, 0.05, 0.1, 0.2, 0.3]
    ]
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for subsample, eta in gridsearch_params:
        print("CV with subsample={}, eta={}".format(
                                 subsample,
                                 eta))
        # Update our parameters
        params['subsample'] = subsample
        params['eta'] = eta
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample,eta)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
    return best_params[0], best_params[1], min_mae
    #Best params: 1.0, 0.6, MAE: 98.3690782

def gamma_tuning(params, dtrain, dtest):

    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for gamma in [i/10. for i in range(0,5)]:
        print("CV with gamma={}".format(gamma))
        # Update our parameters
        params['gamma'] = gamma
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = gamma
    print("Best params: {}, MAE: {}".format(best_params, min_mae))
    return best_params, min_mae
    #Best params: 1.0, 0.6, MAE: 98.3690782

def hyperparameter_tuning():
    region = 'quick'
    tag = '300'
    settings = [False, False, '', 'onehot', 'hypertuning']

    data, labels = import_region_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {
        # Parameters that we are going to tune.
        'max_depth':6,
        'min_child_weight': 2,
        'eta':.01,
        'subsample': 1,
        'colsample_bytree': 0.6,
        'gamma':0.2,
        # Other parameters
        'objective':'reg:squarederror',
    }
    eta_vals = [0.01, 0.05, 0.1, 0.2, 0.3]
    params['eval_metric'] = "mae"
    gridsearch_params = [
        (max_depth, min_child_weight, subsample, colsample, gamma, eta)
        for max_depth in range(2,12,2)
        for min_child_weight in range(1,6)
        for subsample in [i/10. for i in range(6,11)]
        for colsample in [i/10. for i in range(6,11)]
        for gamma in [i/10. for i in range(0,5)]
        for eta in eta_vals
    ]

    scores = ['mse', 'r2']
    # Define initial best params and MAE
    min_mae = float("Inf")
    best_params = None

    for max_depth, min_child_weight, subsample, colsample, gamma, eta in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}, subsample={}, colsample={}, gamma={}, eta={}".format(
                                 max_depth,
                                 min_child_weight,
                                 subsample,
                                 colsample,
                                 gamma,
                                 eta))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        params['subsample'] = subsample
        params['colsample'] = colsample
        params['gamma'] = gamma
        params['eta'] = eta

        start = time.process_time()
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=451,
            seed=42,
            nfold=5,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} time taken {}\n".format(mean_mae, boost_rounds, time.process_time() - start))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = params
    print("Best params: max_depth={}, min_child_weight={}, subsample={}, colsample={}, gamma={}, eta={}, MAE: {}".format(best_params['max_depth'], best_params['min_child_weight'], best_params['subsample'], best_params['colsample'], best_params['gamma'], best_params['eta'], min_mae))

    #Best params: 0.01  (eta), MAE: 98.04678799999999
    #Best params: 0 (gamma), MAE: 98.3484922

def hyperparameter_tuning_rf(data, labels):

    settings = [False, False, '', 'onehot', 'hypertuning']

    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(x_train, y_train)
    base_accuracy = evaluate(base_model, x_test, y_test)

    n_estimators=[int(i) for i in range(100, 3100, 100)]
    max_features = [1/6, 1/3, 1/2]
    param_grid = {
        # Parameters that we are going to tune.
        'n_estimators':n_estimators,
        'max_features': max_features,
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model

    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 2)

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, x_test, y_test)
    print(f'Best grid {best_grid}')

    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    #Best params: 0.01  (eta), MAE: 98.04678799999999
    #Best params: 0 (gamma), MAE: 98.3484922

def tune_xgboost(x_train, x_test, y_train, y_test):

    params = {
        # Parameters that we are going to tune.
        'max_depth':10,
        'min_child_weight': 3,
        'gamma': 0.2,
        'eta':.05,
        'subsample': 0.8,
        # Other parameters
        'objective':'reg:squarederror',
    }

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    start = time.process_time()
    max_depth,min_child_weight, mae = xgb_maxdepth_childweight(params, dtrain, dtest)
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    subsample, eta, mae = eta_sub_tuning(params, dtrain, dtest)
    params['subsample'] = subsample
    params['eta'] = eta
    gamma, mae = gamma_tuning(params, dtrain, dtest)
    params['gamma'] = gamma
    print('Taken {} time'.format(time.process_time()-start))
    return params, mae

def tune_all_xgboost():
    print('Tuning xgboost for feature eng')
    region = 'quick'
    tag = '300'
    settings = [True, True, 'pda', 'onehot', 'hypertuning']

    data, labels = import_region_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)

    params = {
        # Parameters that we are going to tune.
        'max_depth':8,
        'min_child_weight': 4,
        'gamma': 0.2, #gamma:0
        'eta':.05, #eta: 0.01
        'subsample': 0.8, #subsample: 0.6
        # Other parameters
        'objective':'reg:squarederror',
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    subsample, eta, mae = eta_sub_tuning(params, dtrain, dtest)
    params['subsample'] = subsample
    params['eta'] = eta
    gamma, mae = gamma_tuning(params, dtrain, dtest)
    params['gamma'] = gamma
    print("Best params COMBINATION: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
        params['max_depth'], params['min_child_weight'], params['subsample'], gamma, params['eta'], mae))


    data, labels = import_sentinel_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    params, mae = tune_xgboost(x_train, x_test, y_train, y_test)
    print("Best params SENTINEL 1A: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
        params['max_depth'], params['min_child_weight'], params['subsample'], params['gamma'], params['eta'], mae))

    data, labels = import_landsat_grid(region, tag)
    labels = labels.values.ravel()
    x_train, x_test, y_train, y_test = data_transformation(data, labels, settings)
    params, mae = tune_xgboost(x_train, x_test, y_train, y_test)
    print("Best params LANDSAT8: max_depth={}, min_child_weight={}, subsample={}, gamma={}, eta={}, MAE: {}".format(
        params['max_depth'], params['min_child_weight'], params['subsample'], params['gamma'], params['eta'], mae))
