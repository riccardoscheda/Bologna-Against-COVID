#!/usr/bin/env python
# coding: utf-8

import sys
import datetime

import pandas as pd
import numpy as np

import tensorflow as tf

from tqdm import tqdm

import matplotlib.pyplot as plt

class NetSEIR(tf.keras.Model):

    def __init__(self, n_input, n_hidden, S_0, E_0, I_0, R_0, N, LR=1e7):

        self.S_0 = S_0
        self.E_0 = E_0
        self.I_0 = I_0
        self.R_0 = R_0

        self.e = tf.Variable(1e-3, dtype='float64')

        self.S_ = tf.Variable([self.S_0], dtype='float64')
        self.E_ = tf.Variable([self.E_0 * self.e], dtype='float64')
        self.I_ = tf.Variable([self.I_0], dtype='float64')
        self.I_c = tf.Variable([self.I_0], dtype='float64')
        self.R_ = tf.Variable([self.R_0], dtype='float64')
        self.N = tf.Variable(N, dtype='float64')

        self.n_input = n_input
        n_hidden = 8
        self.initializer = tf.random_uniform_initializer(seed=42)

        self.w_beta = tf.Variable(self.initializer(shape=(self.n_input, n_hidden), dtype='float64'), trainable=True)
        self.w_beta_1 = tf.Variable(self.initializer(shape=(n_hidden,), dtype='float64'), trainable=True)
        # self.w_beta_2 = tf.Variable(self.initializer(shape=(self.n_input, ), dtype='float64'), trainable=True)
        # # self.w1_beta = tf.Variable(self.initializer_beta(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w2_beta = tf.Variable(self.initializer_beta(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w3_beta = tf.Variable(self.initializer_beta(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w4_beta = tf.Variable(self.initializer_beta(shape=(self.n_hidden,), dtype='float64'), trainable=True)
        # self.initializer_gamma = tf.random_uniform_initializer(seed=42)
        self.w_gamma = tf.Variable(self.initializer(shape=(self.n_input, n_hidden), dtype='float64'), trainable=True)
        self.w_gamma_1 = tf.Variable(self.initializer(shape=(n_hidden,), dtype='float64'), trainable=True)
        # self.w_gamma_2 = tf.Variable(self.initializer(shape=(self.n_input, ), dtype='float64'), trainable=True)
        # # self.w1_gamma = tf.Variable(self.initializer_gamma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w2_gamma = tf.Variable(self.initializer_gamma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w3_gamma = tf.Variable(self.initializer_gamma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w4_gamma = tf.Variable(self.initializer_gamma(shape=(self.n_hidden,), dtype='float64'), trainable=True)
        # self.initializer_sigma = tf.random_uniform_initializer(seed=42)
        self.w_sigma = tf.Variable(self.initializer(shape=(self.n_input, n_hidden), dtype='float64'), trainable=True)
        self.w_sigma_1 = tf.Variable(self.initializer(shape=(n_hidden,), dtype='float64'), trainable=True)
        # self.w_sigma_2 = tf.Variable(self.initializer(shape=(self.n_input, ), dtype='float64'), trainable=True)
        # # self.w1_sigma = tf.Variable(self.initializer_sigma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w2_sigma = tf.Variable(self.initializer_sigma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w3_sigma = tf.Variable(self.initializer_sigma(shape=(self.n_hidden, self.n_hidden), dtype='float64'), trainable=True)
        # # self.w4_sigma = tf.Variable(self.initializer_sigma(shape=(self.n_hidden,), dtype='float64'), trainable=True)
        #
        self.LR = LR
        self.lambda_beta = tf.Variable(0, dtype='float64')
        self.violatin_beta = tf.Variable(0, dtype='float64')
        self.lambda_gamma = tf.Variable(0, dtype='float64')
        self.violatin_gamma = tf.Variable(0, dtype='float64')
        self.lambda_sigma = tf.Variable(0, dtype='float64')
        self.violatin_sigma = tf.Variable(0, dtype='float64')

    def __step(self, t):
        beta = tf.nn.relu(tf.tensordot(self.X[t], self.w_beta, 1))
        beta = tf.nn.sigmoid(tf.tensordot(beta, self.w_beta_1, 1))
        # beta = tf.nn.tanh(tf.tensordot(beta, self.w_beta_2, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w1_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w2_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w3_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w4_beta, 1))
        # beta = tf.Variable(.1, dtype='float64')
        gamma = tf.nn.relu(tf.tensordot(self.X[t], self.w_gamma, 1))
        gamma = tf.nn.sigmoid(tf.tensordot(gamma, self.w_gamma_1, 1))
        # gamma = tf.nn.tanh(tf.tensordot(gamma, self.w_gamma_2, 1))
        # gamma = tf.tensordot(gamma, self.w1_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w2_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w3_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w4_gamma, 1)
        # gamma = tf.constant(.1, dtype='float64')
        sigma = tf.nn.relu(tf.tensordot(self.X[t], self.w_sigma, 1))
        sigma = tf.nn.sigmoid(tf.tensordot(sigma, self.w_sigma_1, 1))
        # sigma = tf.nn.tanh(tf.tensordot(sigma, self.w_sigma_2, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w1_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w2_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w3_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w4_sigma, 1))
        # sigma = tf.constant(.1, dtype='float64')
        S_to_E = (beta * self.I_c[t] * self.S_[t]) / self.N
        E_to_I = sigma * self.E_[t]
        I_to_R = gamma * self.I_c[t]
        self.S_ = tf.concat([self.S_, [self.S_[t] - S_to_E]], 0)
        self.E_ = tf.concat([self.E_, [self.E_[t] + S_to_E - E_to_I]], 0)
        self.I_ = tf.concat([self.I_, [E_to_I]], 0)
        self.I_c = tf.concat([self.I_c, [self.I_[t] + E_to_I - I_to_R]], 0)
        self.R_ = tf.concat([self.R_, [self.R_[t] + I_to_R]], 0)

    def run_simulation(self):
        self.S_ = tf.Variable([self.S_0], dtype='float64')
        self.E_ = tf.Variable([self.N * self.e], dtype='float64')
        self.I_ = tf.Variable([self.I_0], dtype='float64')
        self.I_c = tf.Variable([self.I_0], dtype='float64')
        self.R_ = tf.Variable([self.R_0], dtype='float64')
        for t in range(self.X.shape[0]-1):
            self.__step(t)

    def loss_op(self):
        self.run_simulation()
        betas = tf.nn.relu(tf.tensordot(self.X, self.w_sigma, 1))
        betas = tf.nn.sigmoid(tf.tensordot(betas, self.w_sigma_1, 1))
        gammas = tf.nn.relu(tf.tensordot(self.X, self.w_gamma, 1))
        gammas = tf.nn.sigmoid(tf.tensordot(gammas, self.w_gamma_1, 1))
        sigmas = tf.nn.relu(tf.tensordot(self.X, self.w_sigma, 1))
        sigmas = tf.nn.sigmoid(tf.tensordot(sigmas, self.w_sigma_1, 1))
        self.run_simulation()
        beta_ = betas[0:-1]
        beta__ = betas[1:]
        self.violatin_beta = (beta_ - beta__) ** 2
        gamma_ = gammas[0:-1]
        gamma__ = gammas[1:]
        self.violatin_gamma = (gamma_ - gamma__) ** 2
        sigma_ = sigmas[0:-1]
        sigma__ = sigmas[1:]
        self.violatin_sigma = (sigma_ - sigma__) ** 2
        # print("loss -: " + str(tf.reduce_mean((self.I_ - self.y)**2)))
        return tf.reduce_mean((self.I_ - self.y)**2) +\
               self.lambda_beta * tf.reduce_mean((beta_ - beta__)**2) +\
               self.lambda_gamma * tf.reduce_mean((gamma_ - gamma__)**2) +\
               self.lambda_sigma * tf.reduce_mean((sigma_ - sigma__)**2)

    def optimize_lagrangian_beta(self):
        self.lambda_beta = self.lambda_beta + self.LR * self.violatin_beta

    def optimize_lagrangian_gamma(self):
        self.lambda_gamma = self.lambda_gamma + self.LR * self.violatin_gamma

    def optimize_lagrangian_sigma(self):
        self.lambda_sigma = self.lambda_sigma + self.LR * self.violatin_sigma

    def train(self, X, y, optimizer):
        self.X = tf.Variable(X, dtype='float64')
        self.y = tf.Variable(y, dtype='float64')
        optimizer.minimize(self.loss_op, var_list=[self.w_beta, self.w_sigma, self.w_gamma,
                                                   self.w_beta_1, self.w_sigma_1, self.w_gamma_1,
                                                   # self.w_beta_2, self.w_sigma_2, self.w_gamma_2,
                                                   self.e])
        self.optimize_lagrangian_beta()
        self.optimize_lagrangian_gamma()
        self.optimize_lagrangian_sigma()

    def get_SEIR_fit(self):
        self.run_simulation()
        return self.S_, self.E_, self.I_, self.R_

    def get_SEIR_param(self):
        beta = tf.nn.relu(tf.tensordot(self.X, self.w_beta, 1))
        beta = tf.nn.sigmoid(tf.tensordot(beta, self.w_beta_1, 1))
        # beta = tf.nn.tanh(tf.tensordot(beta, self.w_beta_2, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w1_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w2_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w3_beta, 1))
        # beta = tf.nn.relu(tf.tensordot(beta, self.w4_beta, 1))
        # beta = tf.Variable(.1, dtype='float64')

        gamma = tf.nn.relu(tf.tensordot(self.X, self.w_gamma, 1))
        gamma = tf.nn.sigmoid(tf.tensordot(gamma, self.w_gamma_1, 1))
        # gamma = tf.nn.tanh(tf.tensordot(gamma, self.w_gamma_2, 1))
        # gamma = tf.tensordot(gamma, self.w1_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w2_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w3_gamma, 1)
        # gamma = tf.tensordot(gamma, self.w4_gamma, 1)
        # gamma = tf.constant(.1, dtype='float64')

        sigma = tf.nn.relu(tf.tensordot(self.X, self.w_sigma, 1))
        sigma = tf.nn.sigmoid(tf.tensordot(sigma, self.w_sigma_1, 1))
        # sigma = tf.nn.tanh(tf.tensordot(sigma, self.w_sigma_2, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w1_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w2_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w3_sigma, 1))
        # sigma = tf.nn.relu(tf.tensordot(sigma, self.w4_sigma, 1))
        # sigma = tf.constant(.1, dtype='float64')

        return beta.numpy(), gamma.numpy(), sigma.numpy()

    # def save_weights(self, filename):
    #     tf.io.write_file(filename, self.w_beta)
    #     tf.io.write_file(filename, self.w_beta_1)


def mov_avg(df, window=7, col="NewCases"):
    """Returns the dataset with the moving average col for new cases
    """

    MA = pd.Series(dtype=np.float64)
    for geo in df.GeoID.unique():
        MA = MA.append(df[df["GeoID"] == geo][col].rolling(window=window).mean()).fillna(0)
    df["MA"] = MA

    return df

if __name__ == '__main__':
    import pickle
    import numpy as np
    import pandas as pd
    import sys, os
    import urllib.request
    from sklearn import preprocessing
    # from os.path import pardir, sep
    sys.path.insert(1, '/' + os.path.join(*os.getcwd().split('/')[:-2]) + '/utils')
    from df_preparation import *

    DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
    # Local file
    DATA_FILE = 'data/OxCGRT_latest.csv'


    # if not os.path.exists('data'):
    #     os.mkdir('data')
    # urllib.request.urlretrieve(DATA_URL, DATA_FILE)

    # Load historical data from local file
    df = pd.read_csv(DATA_FILE,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str,
                            "RegionCode": str},
                     error_bad_lines=False)

    # print(df[df['CountryName'] == 'Italy'][['Date', 'ConfirmedCases']])
    # sys.exit()

    # For testing, restrict training data to that before a hypothetical predictor submission date
    HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-06-20")
    df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    df = mov_avg(df)
    # Keep only columns of interest
    id_cols = ['CountryName',
               'RegionName',
               'GeoID',
               'Date']
    cases_col = ['NewCases', 'MA']
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
    df = df[id_cols + cases_col + npi_cols]

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))

    df_italy = df[df['CountryName'] == 'Italy']

    N = 60000000
    I = np.array(df_italy['MA'].replace(0, 1))
    E_0 = 1
    S = N - I - E_0
    X = np.array(df_italy[npi_cols])
    epochs = 2
    model = NetSEIR(len(X[0]), 16, S[0], E_0, I[0], 0, N)
    # model.run_simulation()
    optimizer = tf.compat.v1.train.AdamOptimizer()

    for epoch in tqdm(range(epochs)):
        model.train(X, I, optimizer)

    S_, E_, I_, R_ = model.get_SEIR_fit()
    beta, gamma, sigma = model.get_SEIR_param()
    # model.save_weights('prova')
    # w_b, w_g, w_s = model.get_params()
    # betas = model.get_betas()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(range(len(I)), I, label='target')
    ax1.plot(range(len(I_)), I_, label='fitted')
    ax2.plot(range(len(beta)), beta, label='beta')
    ax2.plot(range(len(gamma)), gamma, label='gamma')
    ax2.plot(range(len(sigma)), sigma, label='sigma')
    plt.legend()
    plt.show()