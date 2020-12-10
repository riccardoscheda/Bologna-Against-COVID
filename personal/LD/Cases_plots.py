# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# %%

db = pd.read_csv("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv",
                 low_memory=False)

db.sort_values(by=['CountryName', 'RegionName', 'Date'])


# %%

def cases_plots(name, 
                region_name='',
                show_moving_average=True,
                days_for_average=7,
                weights='default',
                show_deaths=False,
                show_total=False):

    
    state = db[db['CountryName']==name]
    
    if region_name != '':
        state = state[state['RegionName']==region_name]
        region_name = ' ('+region_name+') '
    
    day_cases = np.ediff1d(state['ConfirmedCases'])
    plt.plot(state['Date'][:-1].astype(str), day_cases, 'b', label='daily cases')
    
    if show_deaths == True:
        day_deaths = np.ediff1d(state['ConfirmedDeaths'])
        plt.plot(state['Date'][:-1].astype(str), day_deaths, 'grey', label='daily deaths')
    
    if show_moving_average == True:
        
        convolve_mode='valid'
        
        # remember that weights are flipped, they go from last to first
        if weights=='default':
            weights = np.ones(days_for_average)/days_for_average
        else:
            weights = np.asarray(weights)
        
        if np.sum(weights) != 1.:
            weights = weights/np.sum(weights)
        
        assert(len(weights)==days_for_average)
        
        avg_cases = np.convolve(day_cases, weights, convolve_mode)
        plt.plot(state['Date'][int(days_for_average/2):-int(days_for_average/2)-days_for_average%2].astype(str), avg_cases, 'r', label=str(days_for_average)+'-days moving average cases')

    
        if show_deaths == True:
            avg_deaths = np.convolve(day_deaths, weights, convolve_mode)
            plt.plot(state['Date'][int(days_for_average/2):-int(days_for_average/2)-days_for_average%2].astype(str), avg_deaths, 'k', label=str(days_for_average)+'-days moving average deaths')

    
    if show_total == True:
          
        plt.plot(state['Date'][:].astype(str), state['ConfirmedCases'], label='total cases')
        
        if show_deaths == True:
            plt.plot(state['Date'][:].astype(str), state['ConfirmedDeaths'], label='total deaths')
    

    
    final_pos, final_tic = [], []
    
    for pos, tic in zip(plt.xticks()[0], plt.xticks()[1]):
        if pos%30==0:
            final_pos.append(pos)
            final_tic.append(str(state.Date.iloc[pos])[4:6]+'/'+str(state.Date.iloc[pos])[6:])
            
            
    plt.xticks(final_pos, final_tic)
    
    plt.xlabel("Day (MM/DD)")
    plt.ylabel("Counts")
    plt.title(name+region_name+" Covid-19 cases")
    plt.grid()
    plt.legend()
    
    plt.show()
    
    
    
# %%

cases_plots(name='United Kingdom',  # Use capital letter for each distinct word and space between them
            region_name='England',
#            weights=[1, 1, 1, 100, 100, 1, 1, 1],  # if you want to use your own weights
            days_for_average=7)  # suggestion: use an odd number to have the sliding window "centered" over a single day

