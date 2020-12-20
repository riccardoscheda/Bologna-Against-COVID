#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso,RandomForestRegressor
from xgboost import XGBCRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import mae, create_dataset, skl_format
from utils import add_temp, add_population_data, add_HDI
from utils import create_model
from utils import data_to_timesteps

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

scorers = {
    # 'precision_score': make_scorer(precision_score),
    # 'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
    }

id_cols = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
cases_col = ['NewCases']
npi_cols = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, filename='logfile', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    logging.info('################### TRAINING ##################')
    # logging.captureWarnings(True)

    # Reads info from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonfile',
                        dest='JSONfilename',
                        help='JSON configuration file',
                        metavar='FILE',
                        default='config.json')
    args = parser.parse_args()

    print('Loading', args.JSONfilename, '...')
    logging.info('Loading ' + str(args.JSONfilename) + '...')
    with open(args.JSONfilename) as f:
        config_data = json.load(f)

    # Loading config parameters
    lookback_days = config_data['lookback_days']
    train_config = config_data['train']

    input_dataset = train_config['input_file']
    models_output_dir = train_config['models_output_dir']

    start_date = train_config['start_date']
    end_date = train_config['end_date']

    moving_average = eval(train_config['moving_average'])  # it's a string in json, we want bool
    models = train_config['models']
    countries = train_config['countries']

    # Additional Columns adder
    adj_cols_fixed = config_data['adj_cols_fixed']
    adj_cols_time = config_data['adj_cols_time']

    # Reading file with historical interventions
    start = time()
    df = pd.read_csv(input_dataset,
                     parse_dates=['Date'],
                     encoding='ISO-8859-1',
                     dtype={'RegionName': str,
                            'RegionCode': str},
                     error_bad_lines=True)

    # Selecting choosen time period from config file
    df = df[(df.Date > start_date) & (df.Date < end_date)]
    df = create_dataset(df)

    # Selecting countries of interest from config file
    # TO TEST ALL COUNTRY, WRITE "countries" : "" in jsonfile
    if countries:
        cols = countries
    else:
        cols = list(df.CountryName.unique())

    new_df = pd.DataFrame()

    for col in cols:
        new_df = new_df.append(df[df.CountryName == col])

    # Formatting data for scikitlearn / keras
    X_samples, y_samples = skl_format(new_df,
                                      moving_average,
                                      lookback_days=lookback_days,
                                      adj_cols_fixed=adj_cols_fixed,
                                      adj_cols_time=adj_cols_time,
                                      )

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                        y_samples,
                                                        test_size=0.2,
                                                        random_state=301)

    # Start looping on models keys, every model cointains: name and param_grid
    for model_name in models.keys():

        # Flag for LSTM model
        if model_name == "LSTM()":

            model = KerasRegressor(build_fn=create_model, verbose=0)
            models[model_name]['lookback_days'] = f'[{lookback_days}]'

            n_features = len(cases_col + npi_cols + adj_cols_fixed + adj_cols_time)
            models[model_name]['features'] = f'[{n_features}]'

            # Reshape to make Keras Happy
            X_samples = X_samples.reshape(-1,n_features,lookback_days).transpose(0,2,1)
            X_test = X_test.reshape(-1,n_features,lookback_days).transpose(0,2,1)
            X_train = X_train.reshape(-1,n_features,lookback_days).transpose(0,2,1)

        else:
            model = eval(model_name)

        param_grid = models[model_name]

        for param in models[model_name]:
          param_grid[param] = eval(param_grid[param])

        gcv = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='neg_mean_absolute_error',     
                           n_jobs=1,         # -1 is ALL PROCESSOR AVAILABLE
                           cv=2,             # None is K=5 fold CV
                           refit=True,
                           )

        # Fit the GridSearch
        gcv.fit(X_samples, y_samples)

        # Evaluate model
        # train_preds = gcv.predict(X_test)
        # train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases
        # print('\nTrain MAE:', mae(train_preds, y_train))

        test_preds = gcv.predict(X_test)
        test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
        print('Test MAE:', mae(test_preds, y_test))

        model_path = os.path.join(models_output_dir, model_name[:-2] + '.pkl')

        print('Saving model in ', model_path)
        logging.info('Saving model in ' + str(model_path))

        # Save model to file
        if not os.path.exists(models_output_dir):
            os.mkdir(models_output_dir)

        # Take a look at the best params
        print(gcv.best_params_)

        if model_name == 'LSTM()':
            model_path = model_path.split('.')[0] + '.h5'
            gcv.best_estimator_.model.save(model_path)

        else:
            with open(model_path, 'wb') as model_file:
                pickle.dump(gcv, model_file)

        print('Elapsed time: {:.5} s'.format(time() - start))
        logging.info('Elapsed time: ' + str(time() - start))
