import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm
import time
import csv
import sys, os
#from os.path import pardir, sep 
sys.path.insert(1,'/'+os.path.join(*os.getcwd().split('/')[:-2]))
from utils.custom_models import SIR_fitter, SIR_predictor
from pipeline.utils import *

DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
# Local file
DATA_FILE = 'data/OxCGRT_latest.csv'

df = pd.read_csv(DATA_FILE, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)

HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")
df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]
df=create_dataset(df,drop=False)

id_cols = ['CountryName',''
           'RegionName',
           'GeoID',
           'Date']
# Columns we care just about the last value (usually it's always the same value for most of them)
adj_cols_fixed=[ 'Population']#,
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

df.update(df.groupby('GeoID').ConfirmedCases.apply(
    lambda group: group.interpolate()).fillna(0))
df = df[id_cols+ cases_col  +adj_cols_fixed+ adj_cols_time+ npi_cols]
df.loc[df.MA<0,'MA']=0.

lookback_days=30

X_samples, y_samples= skl_format(df,True,lookback_days,adj_cols_fixed,adj_cols_time,
                                True)
X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                    y_samples,
                                                    test_size=0.2,
                                                    random_state=301)

models_str=["MultiTaskLassoCV(alphas=[1e-5,1e-7,1e-9] normalize=True,  max_iter=500000, tol=1e-4,  cv=3, verbose=False, n_jobs=14,selection='random')"]
res_list=[]
s_t=time.time()
i=0
for lookback_days in [15,30]:
    for infection_days in [7,10,15]:
        for semi_fit_days in [4,7,10]:
            for md_str in models_str:
                print('\n',[lookback_days,infection_days, semi_fit_days,
                                md_str])
                i=i+1
                SP=SIR_predictor(df,moving_average=True,lookback_days=lookback_days,
                     infection_days=7,ML_model=md_str,
                     semi_fit_days=7,nprocs=26)
                try:
                    SP.fit(X_train,y_train);
                    TMAE=SP.TMAE
                except Exception as e: 
                    print(e)
                    TMAE=np.nan
                print('Training error MAE:',TMAE)
                if i==0:
                    with open('data/SIRfitter_MAE.csv','w') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(['iteration','lookback_days',
                                                   'infection_days',
                                                  'semi_fit_days','md_str',
                                                  'TMAE'])
                        writer.writerow([i,lookback_days,infection_days, semi_fit_days,
                                md_str,TMAE])
                else:
                    with open('data/SIRfitter_MAE.csv','w') as fd:
                        writer = csv.writer(fd)
                        writer.writerow([i,lookback_days,infection_days, 
                                         semi_fit_days, md_str,TMAE])
e_t=time.time()-s_t
print('Elapsed time: {} min'.format(e_t/60))