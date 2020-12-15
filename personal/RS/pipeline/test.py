#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
from time import time

import pandas as pd

import plot
from predict import my_predict_df
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
    logging.info('################ TESTING ###################')
    logging.captureWarnings(True)

    # reads info from configuration file
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
    test_config = config_data['test']

    input_dataset = test_config['input_file']
    output_dataset = test_config['output_file']

    start_date = test_config['start_date']
    end_date = test_config['end_date']

    moving_average = eval(test_config['moving_average'])  # it's a string in json, we want bool
    models_input_files = test_config['models_input_files']
    countries = test_config['countries']

    # Additional Columns adder
    adj_cols_fixed = config_data['adj_cols_fixed']
    adj_cols_time = config_data['adj_cols_time']

    start = time()

    # Making predictions of choosen countries and saving
    tot = pd.DataFrame()
    for model in models_input_files:
        if "LSTM" in model:
            # TODO: Add an lstm model different from xprize one
            print("selected LSTM model...")
            predictor = XPrizePredictor(model, test_config["input_file"])
            print("predicting...")
            preds_df = predictor.predict(start_date, end_date, test_config["input_file"])
            preds_df['Model'] = model.split(os.sep)[-1].split('.')[0]

            print("finished")
        else:
            preds_df = my_predict_df(countries,
                                     start_date,
                                     end_date,
                                     lookback_days,
                                     moving_average=moving_average,
                                     adj_cols_time=adj_cols_time,
                                     adj_cols_fixed=adj_cols_fixed,
                                     path_to_ips_file=input_dataset,
                                     model_input_file=model,
                                     verbose=False
                                     )

            preds_df['Model'] = model.split(os.sep)[-1].split('.')[0]

        tot = tot.append(preds_df)

    tot.to_csv(output_dataset, index=False)
    print('Saved to ' + output_dataset)
    logging.info('Saved to ' + output_dataset)

    print('Plotting in plot.html')
    logging.info('Plotting in plot.html')

    plot.covid_plot(input_dataset, output_dataset)
    print('Elapsed time: {:.5} s'.format(time() - start))
    logging.info('Elapsed time:' + str(time() - start))
