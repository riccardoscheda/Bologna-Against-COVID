import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm
import time
import sys, os
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#from os.path import pardir, sep 
sys.path.insert(1,'/'+os.path.join(*os.getcwd().split('/')[:-2]))
from pipeline.custom_models import SIR_fitter, SIR_predictor
from pipeline.utils import *

# Main source for the training data
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Local file
DATA_FILE = 'data/OxCGRT_latest.csv'

df = pd.read_csv(DATA_FILE, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)

HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-10-15")
df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

df=create_dataset(df,drop=False)

# Keep only columns of interest
id_cols = ['CountryName',''
           'RegionName',
           'GeoID',
           'Date']
# Columns we care just about the last value (usually it's always the same value for most of them)
adj_cols_fixed=['ConfirmedCases', 'Population']#,
       #'Population Density (# per km2)',
       #'Urban population (% of total population)',
       #'Population ages 65 and above (% of total population)',
       #'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)',
       #'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
       #'Share of Deaths from Air Pollution (%)',
       #'CO2 emissions (metric tons per capita)',
       #'Air transport (# carrier departures worldwide)']

# Columns we would like to include for the last nb_lookback days
adj_cols_time=['TemperatureC']


cases_col = ['MA']
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
df = df[id_cols+ cases_col  +adj_cols_fixed+ adj_cols_time+ npi_cols]

df=df[df['CountryName'].isin(['Italy','Germany','Spain','France'])].sort_values(
    ['GeoID','Date'])

df.loc[df.MA<0,'MA']=0.

lookback_days=30

X_samples, y_samples= skl_format(df,True,lookback_days,adj_cols_fixed,adj_cols_time,
                                True)
print(X_samples.shape)
print(y_samples.shape)

X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                    y_samples,
                                                    test_size=0.2,
                                                    random_state=301)

param_grid={'semi_fit':[3,7],
           'infection_days':[3,7,10].
           'MLmodel__learning_rate':[0.1,0.3]}
gcv = GridSearchCV(estimator=SIR_predictor(df,moving_average=True,lookback_days=lookback_days,infection_days=7,
                 semi_fit=7,nprocs=26),
                           param_grid=param_grid,
                           scoring=None,  # TODO
                           n_jobs=1,      # -1 is ALL PROCESSOR AVAILABLE
                           cv=2,          # None is K=5 fold CV
                           refit=False,
                   verbose=1
                           )

        # Fit the GridSearch
gcv.fit(X_train, y_train);

joblib.dump(gcv, 'models/gcv.pkl')