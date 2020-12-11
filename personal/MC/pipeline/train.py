#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import mae, create_dataset, skl_format, add_temp

# Keep only columns of interest
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

    # Reads info from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonfile',
                        dest='JSONfilename',
                        help='JSON configuration file',
                        metavar='FILE',
                        default='train_config.json')
    args = parser.parse_args()

    print('Loading', args.JSONfilename, '...')
    logging.info('Loading ' + str(args.JSONfilename) + '...')
    with open(args.JSONfilename) as f:
        config_data = json.load(f)

    start = time()

    # Reading file with historical interventions
    df = pd.read_csv(config_data['input_file'],
                     parse_dates=['Date'],
                     encoding='ISO-8859-1',
                     dtype={'RegionName': str,
                            'RegionCode': str},
                     error_bad_lines=True)

    # Adding temperatures
    # df = add_temp(df)

    # Reading the choosen model
    models = config_data['models']

    # Start looping on models keys, every model cointains: name and param_grid
    for model_name in models.keys():
      model = eval(model_name)
      param_grid = models[model_name]

      for param in models[model_name]:
        param_grid[param] = eval(param_grid[param])

      gcv = GridSearchCV(estimator=model,
                         param_grid=param_grid,
                         scoring=None,  # TODO
                         n_jobs=2,      # -1 is ALL PROCESSOR AVAILABLE
                         cv=None,       # None is K=5 fold CV
                         refit=True,
                         )

      # Selecting countries of interest from config file
      if config_data['countries'] != '':
        cols = config_data['countries']

      else:
        cols = list(df['CountryName'].unique())

      new_df = pd.DataFrame()
      for col in cols:
        new_df = new_df.append(df[df['CountryName'] == col])

      # formatting data for scikitlearn
      lookback_days = config_data['lookback_days']
      X_samples, y_samples = skl_format(create_dataset(new_df), lookback_days)

      # Split data into train and test sets
      X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                          y_samples,
                                                          test_size=0.2,
                                                          random_state=42)

      # Fit the GridSearchCV
      gcv.fit(X_train, y_train)

      # Evaluate model
      train_preds = gcv.predict(X_train)
      train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases
      print('Train MAE:', mae(train_preds, y_train))

      # test_preds = model.predict(X_test)
      # test_preds = np.maximum(test_preds, 0) # Don't predict negative cases
      # print('Test MAE:', mae(test_preds, y_test))

      print('Saving model in models/' + model_name[:-2] + '.pkl')
      logging.info('Saving model in models/' + model_name[:-2] + '.pkl')

      # Save model to file
      if not os.path.exists('models'):
        os.mkdir('models')

      with open('models/' + model_name[:-2], 'wb') as model_file:
        pickle.dump(gcv, model_file)

      print('Elapsed time:', time() - start)
      logging.info('Elapsed time:' + str(time() - start))

    # mode
#    mode = 0o666
    # path = os.path.join(parent_dir, directory)
    # output_path='data/'
    # if not os.path.exists(output_path):
    #     os.mkdir(output_pathd)
