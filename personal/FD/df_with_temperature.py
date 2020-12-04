def add_temp(df):
    df_T=pd.read_csv('data/country_temperatures.csv',parse_dates=['Date'])
    country_rename={'Faroe Islands':'Faeroe Islands','Swaziland':'Eswatini','Kyrgyzstan':'Kyrgyz Republic','Slovakia':'Slovak Republic',
               'Myanmar (Burma)':'Myanmar','Congo (Democratic Republic of the)':'Democratic Republic of Congo','Palau':'Guam',
                'Macedonia':'Kosovo','Congo (Republic of the)':'Congo','Korea':'South Korea','Timor Leste':'Timor-Leste'}
    country_copy={'Bermuda':'Bahamas','United States Virgin Islands':'Barbados','Aruba':'Barbados','Serbia':'Bosnia and Herzegovina',
             'San Marino':'Italy','Hong Kong':'China','Taiwan':'China','Macao':'China','Palestine':'Israel'}
    df_T.Country=df_T.Country.map(country_rename).fillna(df_T['Country'])
    df_T=df.merge(df_T,how='left',left_on=['CountryName','Date'],right_on=['Country','Date'])
    for co_to,co_from in country_copy.items():
        try:
            df_T.loc[df_T.CountryName==co_to,'TemperatureC']=df_T.loc[df_T.CountryName==co_from,'TemperatureC'].values
        except ValueError:
            print(co_to)
    df_T.drop(['Country'],axis=1,inplace=True)
    return df_T