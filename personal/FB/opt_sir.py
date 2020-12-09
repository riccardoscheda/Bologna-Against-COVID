#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bayes_opt import BayesianOptimization
from scipy.integrate import odeint


class SIR():
    """ SIR simulator class """
    def __init__(self, S_0, I_0, R_0, N, t, gamma=1/17):
        """ Initialization function

        :param S_0: integer representing the initial number of susceptible in the population
        :param I_0: integer representing the initial number of infected in the population
        :param R_0: integer representing the initial number of recovered in the population
        :param N: integer representing the number of subjects in the population
        :param t: list representing the days span of the simulation
        """
        self.S_0 = S_0
        self.I_0 = I_0
        self.R_0 = R_0
        self.N = N
        self.t = t
        self.gamma = gamma
        # bounds of the combinatorial varaibles
        self.pbounds = {'beta': (0, 1)}
        # optmizer initialization
        self.optimizer = BayesianOptimization(f=self.__objective__,
                                              pbounds=self.pbounds,
                                              random_state=42)

    def __call__(self, beta):
        y = [self.S_0, self.I_0, self.R_0]
        ret = odeint(self.__deriv__, y, self.t, args=(self.N, beta))
        self.S, self.I, self.R = ret.T
        return self.S, self.I, self.R

    def __deriv__(self, y, t, N, beta):
        """ Function to be derived, basically the SIR formulas

        :param y: initial SIR variables
        :param t: list representing the days span of the simulation
        :param N: integer representing the number of subjects in the population
        :param beta: probability of transmitting disease between a susceptible and an infectious
        :param gamma: average duration of the infection
        """
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N
        dRdt = self.gamma * I
        return dSdt, dIdt, dRdt

    def __objective__(self, beta):
        """ Objective function to minimize

        :param beta: probability of transmitting disease between a susceptible and an infectious
        :param gamma: average duration of the infection
        """
        y = [self.S_0, self.I_0, self.R_0]
        ret = odeint(self.__deriv__, y, self.t, args=(self.N, beta))
        self.S, self.I, self.R = ret.T
        # print(- np.mean((self.I - self.I_g)**2) - np.mean((self.R - self.R_g)**2) - 0.4 * (self.I[-1] - self.I[0])**2)
        return - np.mean((self.I - self.I_g)**2) # - np.mean((self.R - self.R_g)**2) - 0.4 * (self.I[-1] - self.I[0])**2

    def optimize(self, S_g, I_g, R_g):
        """ Optimize the target function based on the ground truth

        :param S_g: list(integer) representing the ground truth of the susceptible population
        :param I_g: list(integer) representing the ground truth of the infected population
        :param R_g: list(integer) representing the ground truth of the recovered population
        """
        self.S_g = S_g
        self.I_g = I_g
        self.R_g = R_g
        # run optimization
        self.optimizer.maximize(
            init_points=5,
            n_iter=50,
        )
        return self.optimizer.max['params']


class SI():
    """ SIR simulator class """
    def __init__(self, S_0, I_0, N, t, gamma=1/17):
        """ Initialization function

        :param S_0: integer representing the initial number of susceptible in the population
        :param I_0: integer representing the initial number of infected in the population
        :param N: integer representing the number of subjects in the population
        :param t: list representing the days span of the simulation
        """
        self.S_0 = S_0
        self.I_0 = I_0
        self.N = N
        self.t = t
        self.gamma = gamma
        # bounds of the combinatorial varaibles
        self.pbounds = {'beta': (0, 1),
                        # 'mu': (0,0.5)
                        }
        # optmizer initialization
        self.optimizer = BayesianOptimization(f=self.__objective__,
                                              pbounds=self.pbounds,
                                              random_state=42)

    def __call__(self, beta):
        y = [self.S_0, self.I_0]
        ret = odeint(self.__deriv__, y, self.t, args=(self.N, beta))
        self.S, self.I = ret.T
        return self.S, self.I


    def __deriv__(self, y, t, N, beta):
        """ Function to be derived, basically the SIR formulas

        :param y: initial SIR variables
        :param t: list representing the days span of the simulation
        :param N: integer representing the number of subjects in the population
        :param beta: probability of transmitting disease between a susceptible and an infectious
        """
        S, I = y
        dSdt = -(beta * S * I) / N
        dIdt = (beta * S * I) / N  - (I * self.gamma)
        return dSdt, dIdt

    def __objective__(self, beta):
        """ Objective function to minimize

        :param beta: probability of transmitting disease between a susceptible and an infectious
        """
        y = [self.S_0, self.I_0]
        ret = odeint(self.__deriv__, y, self.t, args=(self.N, beta))
        self.S, self.I = ret.T
        return - np.mean((self.I - self.I_g)**2) - 0.4 * (self.I[-1] - self.I[0])**2

    def optimize(self, S_g, I_g):
        """ Optimize the target function based on the ground truth

        :param S_g: list(integer) representing the ground truth of the susceptible population
        :param I_g: list(integer) representing the ground truth of the infected population
        """
        self.S_g = S_g
        self.I_g = I_g
        # run optimization
        self.optimizer.maximize(
            init_points=5,
            n_iter=50,
        )
        return self.optimizer.max['params']


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    N = 10000
    R = 0
    I = 1
    S = N - I - R
    days = 100
    t = np.linspace(0, days, days)
    model = SIR(S, I, R, N, t)

    beta = 0.2
    S_target, I_target, R_target = model(beta)

    model = SIR(S, I, R, N, t)
    params = model.optimize(S_target, I_target, R_target)
    S, I, R = model(params['beta'])

    plt.plot(range(len(I_target)), I_target, alpha=0.5, label='Target')
    plt.plot(range(len(I)), I, alpha=0.5, label='Optimized')
    plt.title('Infected')
    plt.legend()
    plt.show()

