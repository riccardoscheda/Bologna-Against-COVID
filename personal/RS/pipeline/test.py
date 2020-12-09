import json
import pandas as pd
from pprint import pprint
from time import time
from argparse import ArgumentParser
import os
import logging
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import pickle


import predict
from predict import predict_df
#import train
import plot
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
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("################ TESTING ###################")
    logging.captureWarnings(True)

    #reads info from configuration file
    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonfile",
                        dest="JSONfilename",
                        help="JSON configuration file",
                        metavar="FILE",
                        default="test_config.json")
    args = parser.parse_args()
    print('Loading', args.JSONfilename, '...')
    logging.info('Loading '+  str(args.JSONfilename) + '...')
    with open(args.JSONfilename) as f:
        config_data = json.load(f)

    start = time()

    #making predictions of choosen countries and saving
    countries = config_data["countries"]
    preds_df = predict_df(countries, config_data["start_date"], config_data["end_date"], path_to_ips_file=config_data["input_file"],model_input_file=config_data["model_input_file"], verbose=False)
    preds_df['NewCases'] = preds_df.groupby(["CountryName"]).PredictedDailyNewCases.diff().fillna(0)
    preds_df.to_csv(config_data["output_file"])

    print("Saved to " + config_data["output_file"])
    logging.info("Saved to " + config_data["output_file"])

    #plotting cases
    print("Plotting in plot.html")
    logging.info("Plotting in plot.html")
    plot.covid_plot(config_data["input_file"],config_data["output_file"])
    print("Elapsed time:", time() - start)
    logging.info("Elapsed time:" + str(time() - start))
