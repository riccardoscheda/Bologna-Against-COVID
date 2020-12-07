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
import pylab as plt
import predict
from predict import predict_df

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
    countries = config_data["countries"]
    preds_df = predict_df(countries, config_data["start_date"], config_data["end_date"], path_to_ips_file=config_data["input_file"], verbose=False)
    # for country in countries:
    #     plt.plot(preds_df[preds_df["CountryName"]==country]["Date"],preds_df[preds_df["CountryName"]==country]["PredictedDailyNewCases"])
    # plt.show()
    preds_df.to_csv(config_data["output_file"])
    print("Saved to " + config_data["output_file"])

    print("Plotting in plot.html")
    os.chdir("../")
    os.system("python3 plot.py")
    print("Elapsed time:", time() - start)
    logging.info("Elapsed time:" + str(time() - start))
