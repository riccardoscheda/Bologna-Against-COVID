def add_temp(df):
    '''Use this only on the Oxford dataframe.
    Return the same dataframe with a column temperature taken from data/country_temperatures.csv'''
    import pandas as pd
    df_T=pd.read_csv('data/country_temperatures.csv',parse_dates=['Date'])
    df_T=df.merge(df_T,how='left',left_on=['CountryName','Date'],right_on=['CountryName','Date'])
    return df_T