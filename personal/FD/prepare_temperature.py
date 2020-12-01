import pandas as pd
import numpy as np

def strip_shorten(text):
    try:
        return text.strip()[:3]
    except AttributeError:
        return text
def strip_entry(text):
    try:
        return text.strip()
    except AttributeError:
        return text
    
def datify_2020(month):
    return pd.to_datetime('2020-'+month,format='%Y-%b')

df_temp=pd.read_csv('data/tas_1991_2016.csv',sep=',',usecols=[0,1,2,3],names=['TemperatureC','Year','Month','Country'],header=0,
                   converters={'Month':strip_shorten,'Country':strip_entry})

df_avg=df_temp.groupby(['Country','Month']).aggregate({'TemperatureC': np.mean}).reset_index()
df_avg['MonthlyDate']=df_avg.apply(lambda row: datify_2020(row['Month']), axis=1)
df_avg.sort_values(['Country','MonthlyDate'],inplace=True)

df_piv = df_avg[['Country','MonthlyDate','TemperatureC']].pivot(index='MonthlyDate', columns='Country')
start_date = df_piv.index.min() - pd.DateOffset(day=1)
end_date = df_piv.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='D')
dates.name = 'Date'
df_piv = df_piv.reindex(dates,fill_value=np.nan)
df_piv.loc['2021-01-01',:]=df_piv.loc['2020-01-01',:]
df_piv.set_index(pd.to_datetime(df_piv.index),inplace=True)
df_piv=df_piv.interpolate(method='quadratic')
df_piv.drop(pd.to_datetime('2021-01-01'),inplace=True)
df_piv = df_piv.stack('Country')
df_piv = df_piv.sort_index(level=1)
df_piv = df_piv.reset_index()

df_piv.to_csv('data/country_temperatures.csv',index=False)

start_date = df_piv.Date.min() - pd.DateOffset(day=1)
end_date = df_piv.Date.max() + pd.DateOffset(day=31)
print('Assigned temperatures to countries, indexed from {} to {}'.format(start_date,end_date))