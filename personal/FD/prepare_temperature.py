import pandas as pd
import numpy as np


end_date = pd.to_datetime('2021-12-31')

country_rename={'Faroe Islands':'Faeroe Islands','Swaziland':'Eswatini','Kyrgyzstan':'Kyrgyz Republic','Slovakia':'Slovak Republic',
               'Myanmar (Burma)':'Myanmar','Congo (Democratic Republic of the)':'Democratic Republic of Congo','Palau':'Guam',
                'Macedonia':'Kosovo','Congo (Republic of the)':'Congo','Korea':'South Korea','Timor Leste':'Timor-Leste'}

#To:From
country_copy={'Bermuda':'Bahamas','United States Virgin Islands':'Barbados','Aruba':'Barbados','Serbia':'Bosnia and Herzegovina',
             'San Marino':'Italy','Hong Kong':'China','Taiwan':'China','Macao':'China','Palestine':'Israel'}


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

DATA_FILE = 'data/OxCGRT_latest.csv'
df = pd.read_csv(DATA_FILE, 
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=False)
df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
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
df = df[id_cols + cases_col + npi_cols]
df.update(df.groupby('GeoID').NewCases.apply(
    lambda group: group.interpolate()).fillna(0))
for npi_col in npi_cols:
    df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))



df_temp=pd.read_csv('data/tas_1991_2016.csv',sep=',',usecols=[0,1,2,3],names=['TemperatureC','Year','Month','CountryName'],header=0,
                   converters={'Month':strip_shorten,'CountryName':strip_entry})
df_temp=df_temp[df_temp.Year>=2010]
df_avg=df_temp.groupby(['CountryName','Month']).aggregate({'TemperatureC': np.mean}).reset_index()
df_avg['MonthlyDate']=df_avg.apply(lambda row: datify_2020(row['Month']), axis=1)
df_avg.sort_values(['CountryName','MonthlyDate'],inplace=True)
df_avg['CountryName']=df_avg.CountryName.map(country_rename).fillna(df_avg['CountryName'])
print('Df avg pre copy: ',df_avg.shape)
print(df_avg)

missing_stati=list(set(df.CountryName)-set(df_avg.CountryName))
for stato in missing_stati:
    try:
        sfrom=country_copy[stato]
        df_stato=df_avg[df_avg.CountryName==sfrom].copy()
        df_stato['CountryName']=stato
        df_avg=pd.concat([df_avg,df_stato]).reset_index(drop=True)
    except KeyError:
        print(stato)
#CICLA FOR E CONCATENA TUTTI I NUOVI STATI COPIANDO DA SOPRA IL DF[DF.COUNTRY==FROM]
print('\nDf avg post copy: ',df_avg.shape)
print(df_avg)

missing_stati=list(set(df.CountryName)-set(df_avg.CountryName))
print(missing_stati)

df_avg.sort_values(['CountryName','MonthlyDate'],inplace=True)

df_piv = df_avg[['CountryName','MonthlyDate','TemperatureC']].pivot(index='MonthlyDate', 
                                                                    columns='CountryName')
start_date = df_piv.index.min() - pd.DateOffset(day=1)
dates = pd.date_range(start_date, end_date, freq='D')
dates.name = 'Date'
df_piv = df_piv.reindex(dates,fill_value=np.nan)
df_piv.loc['2021-01-01',:]=df_piv.loc['2020-01-01',:]
df_piv.set_index(pd.to_datetime(df_piv.index),inplace=True)
df_piv=df_piv.interpolate(method='quadratic')
df_piv.drop(pd.to_datetime('2021-01-01'),inplace=True)
df_piv = df_piv.stack('CountryName')
df_piv = df_piv.sort_index(level=1)
df_piv = df_piv.reset_index()
print(df_piv.tail())

df_piv21=df_piv.copy()
df_piv21['Date']=df_piv21['Date']+pd.DateOffset(years=1)

df_piv=pd.concat([df_piv,df_piv21])

df_piv = df_piv.reset_index(drop=True)
print('\nDf temp pre merge:',df_piv)
df_piv[['CountryName','Date','TemperatureC']].to_csv('data/country_temperatures.csv',index=False)



#df_piv.Country=df_piv.Country.map(country_rename).fillna(df_piv['Country'])


#df_temped=df.merge(df_piv,how='left',left_on=['CountryName','Date'],right_on=['Country','Date'])
#for co_to,co_from in country_copy.items():
#    try:
#        df_temped.loc[df_temped.CountryName==co_to,'TemperatureC']= df_temped.loc[df_temped.CountryName==co_from,'TemperatureC'].values
#    except ValueError:
#        print(co_to)
#df_temped.drop(['Country'],axis=1,inplace=True)
#print(df_temped.tail())


#df_temped[['CountryName','Date','TemperatureC']].to_csv('data/country_temperatures.csv',index=False)

start_date = df_piv.Date.min() 
end_date = df_piv.Date.max() 
print('Assigned temperatures to countries, indexed from {} to {}'.format(start_date,end_date))