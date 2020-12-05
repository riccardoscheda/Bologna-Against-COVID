import json
import pandas as pd
from pprint import pprint
from time import time
from argparse import ArgumentParser
import os
import logging
from sklearn.linear_model import LinearRegression

def create_dataset(df):
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    # Add new cases column



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

    parser = ArgumentParser()
    parser.add_argument("-j", "--jsonfile",
                        dest="JSONfilename",
                        help="JSON configuration file",
                        metavar="FILE",
                        required=True)
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

    model.train()
    print("Elapsed time:", time() - start)
    logging.info("Elapsed time:" + str(time() - start))

    # mode
    mode = 0o666
    #path = os.path.join(parent_dir, directory)
    output_path="data/"
    if not os.path.exists(output_path):
        os.mkdir(output_path, mode)
