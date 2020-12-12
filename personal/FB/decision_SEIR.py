#!/usr/bin/env python
# coding: utf-8

import sys
import datetime

import pandas as pd
import numpy as np

import tensorflow as tf

from tqdm import tqdm
from reader import read_SIR

import matplotlib.pyplot as plt

class NetSEIR(tf.keras.Model):

    def __init__(self, X, S, I, N, T, LR=1e9):
        self.X = X
        self.S = tf.Variable(S, dtype='float64')
        self.I = tf.Variable(I, dtype='float64')
        self.N = tf.Variable(N, dtype='float64')
        self.T = T

        self.S_ = tf.Variable([self.S[0]], dtype='float64')
        self.E_ = tf.Variable([0], dtype='float64')
        self.I_ = tf.Variable([self.I[0]], dtype='float64')
        self.R_ = tf.Variable([self.R[0]], dtype='float64')

        self.initializer_beta = tf.random_uniform_initializer(minval=0, seed=42)
        self.w_beta = tf.Variable(self.initializer_beta(shape=(len(self.X[0]),), dtype='float64'), trainable=True)
        self.initializer_gamma = tf.random_uniform_initializer(minval=0, seed=42)
        self.w_gamma = tf.Variable(self.initializer_gamma(shape=(len(self.X[0]),), dtype='float64'), trainable=True)
        self.initializer_sigma = tf.random_uniform_initializer(minval=0, seed=42)
        self.w_sigma = tf.Variable(self.initializer_sigma(shape=(len(self.X[0]),), dtype='float64'), trainable=True)
        # set lagrangian dual parameters
        self.LR = LR
        self.lambda_beta = tf.Variable(0, dtype='float64')
        self.violatin_beta = tf.Variable(0, dtype='float64')
        self.lambda_gamma = tf.Variable(0, dtype='float64')
        self.violatin_gamma = tf.Variable(0, dtype='float64')
        self.lambda_sigma = tf.Variable(0, dtype='float64')
        self.violatin_sigma = tf.Variable(0, dtype='float64')

    def __step(self, t):
        beta = self.X[t] * self.w_beta
        gamma = self.X[t] * self.w_gamma
        sigma = self.X[t] * self.w_sigma
        S_to_E = (beta * self.I_[t] * self.S_[t]) / self.N
        E_to_I = sigma * self.E_[t]
        I_to_R = gamma * self.I_[t]
        self.S_ = tf.concat([self.S_, [self.S_[t] - S_to_E]], 0)
        self.E_ = tf.concat([self.E_, [self.E_[t] + S_to_E - E_to_I]], 0)
        self.I_ = tf.concat([self.I_, [self.I_[t] + E_to_I - I_to_R]], 0)
        self.R_ = tf.concat([self.R_, [self.R_[t] + I_to_R]], 0)

    def run_simulation(self):
        self.S_ = tf.Variable([self.S[0]], dtype='float64')
        self.E_ = tf.Variable([0], dtype='float64')
        self.I_ = tf.Variable([self.I[0]], dtype='float64')
        self.R_ = tf.Variable([self.R[0]], dtype='float64')
        for t in range(self.T-1):
            self.__step(t)

    def loss_op(self):
        self.run_simulation()
        betas = self.X * self.w_beta
        gammas = self.X * self.w_gamma
        sigmas = self.X * self.w_sigma
        beta_ = betas[0:-1]
        beta__ = betas[1:]
        self.violatin_beta = self.D * (beta_ - beta__) ** 2
        gamma_ = gammas[0:-1]
        gamma__ = gammas[1:]
        self.violatin_gamma = self.D * (gamma_ - gamma__) ** 2
        sigma_ = sigmas[0:-1]
        sigma__ = sigmas[1:]
        self.violatin_sigma = self.D * (sigma_ - sigma__) ** 2
        return tf.reduce_mean((self.I_ - self.I)**2) + \
               self.lambda_beta * self.violatin_beta + \
               self.lambda_gamma * self.violatin_gamma + \
               self.lambda_sigma * self.violatin_sigma

    def optimize_lagrangian_beta(self):
        self.lambda_beta = self.lambda_beta + self.LR * self.violatin_beta

    def optimize_lagrangian_gamma(self):
        self.lambda_gamma = self.lambda_gamma + self.LR * self.violatin_gamma

    def optimize_lagrangian_sigma(self):
        self.lambda_sigma = self.lambda_sigma + self.LR * self.violatin_sigma

    def train(self, optimizer):
        optimizer.minimize(self.loss_op, var_list=[self.w_beta, self.w_gamma, self.w_sigma])
        self.optimize_lagrangian_beta()
        self.optimize_lagrangian_gamma()
        self.optimize_lagrangian_sigma()

    # def get_SEIR(self):
    #     self.run_simulation()
    #     return self.S_, self.E_, self.I_, self.R_, self.betas.numpy(), self.gammas.numpy(), self.sigma.numpy()

#     def predict(self, days):
#         S = [self.S_[-1].numpy()]
#         E = [self.E_[-1].numpy()]
#         I = [self.I_[-1].numpy()]
#         R = [self.R_[-1].numpy()]
#         N = self.N.numpy()
#         beta = self.betas[-2].numpy()
#         gamma = self.gammas[-2].numpy()
#         sigma = self.sigma[-2].numpy()
#         for day in range(days):
#             S_to_E = (beta * I[-1] * S[-1]) / N
#             E_to_I = sigma * E[-1]
#             I_to_R = gamma * I[-1]
#             S.append(S[-1] - S_to_E)
#             E.append(E[-1] + S_to_E - E_to_I)
#             I.append(I[-1] + E_to_I - I_to_R)
#             R.append(R[-1] + I_to_R)
#
#         return S, E, I, R
#         #, tf.get_static_value(self.lambda_beta), tf.get_static_value(self.lambda_gamma)
#
#
# def test(S, I, R, N, window_size, k, epochs=200):
#
#     mse = lambda x, y: (sum(x - y)**2)/window_size
#     s_idx = 0
#     e_idx = s_idx + window_size
#     out_idx = len(S)
#     predicted_I = list(I[0:e_idx])
#     error = [0] * len(predicted_I)
#
#     while e_idx + window_size < out_idx:
#         k_w = k[:e_idx-1]
#         optimizer = tf.compat.v1.train.AdamOptimizer()
#         S_w = S[0:e_idx]
#         I_w = I[0:e_idx]
#         R_w = R[0:e_idx]
#         model = NetSEIR(S_w, I_w, R_w, N, len(S_w), k_w)
#         for epoch in tqdm(range(epochs)):
#             model.train(optimizer)
#         S_p, I_p, R_p, _, _, _, _ = model.predict(7)
#         predicted_I += I_p
#         error += list(np.abs(I_p[1:] - I[e_idx:e_idx+7]))
#         s_idx = e_idx + 7
#         e_idx = s_idx + window_size
#
#     return error, predicted_I

if __name__ == '__main__':
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import sys, os

    # from os.path import pardir, sep
    sys.path.insert(1, '/' + os.path.join(*os.getcwd().split('/')[:-2]) + '/utils')
    from df_preparation import *

    DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    # Local file
    DATA_FILE = 'data/OxCGRT_latest.csv'

    import urllib.request

    if not os.path.exists('data'):
        os.mkdir('data')
    # urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    # Load historical data from local file
    df = pd.read_csv(DATA_FILE,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)

    # For testing, restrict training data to that before a hypothetical predictor submission date
    HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")
    df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

    df = create_dataset(df)

    id_cols = ['CountryName',
               'RegionName',
               'GeoID',
               'Date']
    # Columns we care just about the last value (usually it's always the same value for most of them)
    adj_cols_fixed = ['PastCases', 'Population',
                      'Population Density (# per km2)',
                      'Urban population (% of total population)',
                      'Population ages 65 and above (% of total population)',
                      'GDP per capita (current US$)', 'Obesity Rate (%)', 'Cancer Rate (%)',
                      'Share of Deaths from Smoking (%)', 'Pneumonia Death Rate (per 100K)',
                      'Share of Deaths from Air Pollution (%)',
                      'CO2 emissions (metric tons per capita)',
                      'Air transport (# carrier departures worldwide)']

    # Columns we would like to include for the last nb_lookback days
    adj_cols_time = ['TemperatureC']

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

    # Fill also missing CONFIRMED case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').ConfirmedCases.apply(
        lambda group: group.interpolate()).fillna(0))

    df = add_population_data(df)
    df = add_temp(df)

    df['PastCases'] = df.ConfirmedCases.values

    # Keep only columns of interest
    df = df[id_cols + cases_col + adj_cols_fixed + adj_cols_time + npi_cols]

    lookback_days = 30
    infection_days = 15

    X_samples, y_samples = skl_format(df, cases_col, adj_cols_fixed, adj_cols_time, npi_cols, lookback_days)

    X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                        y_samples,
                                                        test_size=0.2,
                                                        random_state=301)