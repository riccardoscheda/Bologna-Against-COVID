#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import logging
import argparse
from time import time
from itertools import product


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


import sys
sys.path.insert(1,'/'+os.path.join(*os.getcwd().split('/')[:-3]))
from pipeline.custom_models import SIR_fitter, SIR_predictor
from pipeline.utils import mae, create_dataset, skl_format
from pipeline.utils import add_temp, add_population_data, add_HDI

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
    drop_columns_with_Nan = eval(config_data['drop_columns_with_Nan'])
    pre_fit_params=eval(config_data['pre_fit_params'])
    keep_df_index=eval(config_data['keep_df_index'])
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
    df = create_dataset(df, drop=drop_columns_with_Nan)

    # Selecting countries of interest from config file
    # TO TEST ALL COUNTRY, WRITE "countries" : "" in jsonfile
    if countries:
        cols = countries
    else:
        cols = list(df.CountryName.unique())

    new_df = pd.DataFrame()

    for col in cols:
        new_df = new_df.append(df[df.CountryName == col])

    # formatting data for scikitlearn
    X_samples, y_samples = skl_format(new_df,
                                      moving_average,
                                      lookback_days=lookback_days,
                                      adj_cols_fixed=adj_cols_fixed,
                                      adj_cols_time=adj_cols_time,
                                      keep_df_index=keep_df_index)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                        y_samples,
                                                        test_size=0.2,
                                                        random_state=301)
    if pre_fit_params: 
        precomp_df_pars= SIR_predictor(new_df,moving_average=True, lookback_days=lookback_days, 
                                       infection_days=7, semi_fit=7, nprocs=26 
                                      ).fit(X_train,y_train).SFmodel.df_pars
    else:
        precomp_df_pars=None

    # Start looping on models keys, every model cointains: name and param_grid
    for model_name in models.keys():

        model = eval(model_name)
        param_grid = models[model_name]

        for param in models[model_name]:
            param_grid[param] = eval(param_grid[param])
        if 'xgb_gamma' in param_grid.keys():
            gammas=param_grid['xgb_gamma']
        else:
            gammas=[0.2]
        if 'xgb_learning_rate' in param_grid.keys():
            larates=param_grid['xgb_learning_rate']
        else:
            lrates=[0.2]
        if 'xgb_max_depth' in param_grid.keys():
            mdeps=param_grid['xgb_max_depth']
        else:
            mdeps=[2]
        if 'xgb_n_estimators' in param_grid.keys():
            nestims=param_grid['xgb_n_estimators']
        else:
            nestims=[10]
        if 'xgb_subsample' in param_grid.keys():
            ssamples=param_grid['xgb_subsample']
        else:
            ssamples=[1]
        
        mcombs=[]
        for comb in product(gammas,lrates,mdeps,nestims,ssamples):
            mcombs.append('MultiOutputRegressor(xgb.XGBRegressor(gamma={},learning_rate={},max_depth={},n_estimators={},subsample={}))'.format(*comb))
    
        param_grid={'semi_fit':[7],
           'infection_days':[7],
           'MLmodel':mcombs}
        gcv=GridSearchCV(estimator=SIR_predictor(df, moving_average=True, lookback_days=lookback_days, 
                                             infection_days=7, semi_fit=7, nprocs=26, 
                                             pre_computed=precomp_df_pars),
                           param_grid=param_grid,
                           scoring=None,  # TODO
                           n_jobs=1,      # -1 is ALL PROCESSOR AVAILABLE
                           cv=2,          # None is K=5 fold CV
                           refit=True,
                   verbose=1
                           )


        # Fit the GridSearch
        gcv.fit(X_train, y_train)

        # Evaluate model
        train_preds = gcv.predict(X_train)
        train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases
        print('\nTrain MAE:', mae(train_preds, y_train))

        # test_preds = model.predict(X_test)
        # test_preds = np.maximum(test_preds, 0) # Don't predict negative cases
        # print('Test MAE:', mae(test_preds, y_test))

        model_path = os.path.join(models_output_dir, model_name[:-2] + '.pkl')

        print('Saving model in ', model_path)
        logging.info('Saving model in ' + str(model_path))

        # Save model to file
        if not os.path.exists(models_output_dir):
            os.mkdir(models_output_dir)

        with open(model_path, 'wb') as model_file:
            pickle.dump(gcv, model_file)

        print('Elapsed time: {:.5} s'.format(time() - start))
        logging.info('Elapsed time: ' + str(time() - start))
