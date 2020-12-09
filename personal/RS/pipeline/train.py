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

# Helpful function to compute mae
def mae(pred, true):
    return np.mean(np.abs(pred - true))

#This Function need to be outside the training process
def create_dataset(df):
    """
    From XPRIZE jupyter, this function merges country and region, fills any missing cases
    and fills any missing pis
    """
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    # Add new cases column
    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)


    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))


    return df

def skl_format(df, lookback_days=30):
    """
    Takes data and makes a formatting for sklearn
    """
    # Create training data across all countries for predicting one day ahead
    X_cols = cases_col + npi_cols
    y_col = cases_col
    X_samples = []
    y_samples = []
    geo_ids = df.GeoID.unique()
    for g in geo_ids:
        gdf = df[df.GeoID == g]
        all_case_data = np.array(gdf[cases_col])
        all_npi_data = np.array(gdf[npi_cols])

        # Create one sample for each day where we have enough data
        # Each sample consists of cases and npis for previous lookback_days
        nb_total_days = len(gdf)
        for d in range(lookback_days, nb_total_days - 1):
            X_cases = all_case_data[d-lookback_days:d]

            # Take negative of npis to support positive
            # weight constraint in Lasso.
            X_npis = -all_npi_data[d - lookback_days:d]

            # Flatten all input data so it fits Lasso input format.
            X_sample = np.concatenate([X_cases.flatten(),
                                       X_npis.flatten()])
            y_sample = all_case_data[d + 1]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples).flatten()

    return X_samples, y_samples



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("################### TRAINING ##################")

    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonfile",
                        dest="JSONfilename",
                        help="JSON configuration file",
                        metavar="FILE",
                        default="train_config.json")
    args = parser.parse_args()

    print('Loading', args.JSONfilename, '...')
    logging.info('Loading '+  str(args.JSONfilename) + '...')
    with open(args.JSONfilename) as f:
        config_data = json.load(f)

    start = time()

    df = pd.read_csv(config_data["input_file"],
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=True)

    model = eval(config_data["model"])

    cols = config_data["countries"]
    new_df = pd.DataFrame()
    for col in cols:
        new_df = new_df.append(df[df["CountryName"] == col])


    X_samples, y_samples = skl_format(create_dataset(new_df))
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                        y_samples,
                                                        test_size=0.2,
                                                        random_state=301)

    #Fit the model
    model.fit(X_train, y_train)

    # Evaluate model
    train_preds = model.predict(X_train)
    train_preds = np.maximum(train_preds, 0) # Don't predict negative cases

    test_preds = model.predict(X_test)
    test_preds = np.maximum(test_preds, 0) # Don't predict negative cases


    print("Saving model in models/model.pkl")
    logging.info("Saving model in models/model.pkl")

    # Save model to file
    if not os.path.exists('models'):
        os.mkdir('models')
    with open('models/'+config_data["model_output_file"], 'wb') as model_file:
        pickle.dump(model, model_file)


    print("Elapsed time:", time() - start)
    logging.info("Elapsed time:" + str(time() - start))

    # mode
#    mode = 0o666
    #path = os.path.join(parent_dir, directory)
    # output_path="data/"
    # if not os.path.exists(output_path):
    #     os.mkdir(output_pathd)
