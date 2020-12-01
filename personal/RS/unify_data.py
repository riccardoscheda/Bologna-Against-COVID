#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os


#We want to format italian data to the challenge format, so we make a Dataframe equal to the one of the challenge.


def to_final(challenge_df,region, district,total_cases,date):
	 """
	 Returns a dataframe in the format of challenge data, including interventions
	 --------------------------------------------
	 Parameters:
	 
	 challenge_df: pandas dataframe from which we copy column names
	 region: string or list of strings of regions
	 district: string or list of strings of districts,
	 total cases: list of daily total cases
	 date: list of dates
	 """
	 final = pd.DataFrame(columns=challenge_df.columns)
	 final["ConfirmedCases"] = total_cases
	 final["Date"] = date
	 final["CountryName"] ="Italy"
	 final["CountryCode"] = "ITA"
	 final["RegionName"] = region
	 final["DistrictName"] = district
	 return final

# Challenge dataframe
print("Downloading challenge data...")
DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
challenge_df = pd.read_csv(DATA_URL,
                 parse_dates=['Date'],
                 encoding="ISO-8859-1",
                 dtype={"RegionName": str,
                        "RegionCode": str},
                 error_bad_lines=True)
print("loaded")
print("REGIONS:")
print("Downloading data from Protezione Civile...")
# REGIONS #######
file = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
df = pd.read_csv(file)
regions = list(df["denominazione_regione"].unique())
regions_df = pd.DataFrame()

for region in regions:
    print(region + "...                                  ", end="\r")
    total_cases = df[df["denominazione_regione"] == region]
    em_df = total_cases["totale_casi"].values
    date = total_cases["data"].values

    for i in range(len(date)):
        date[i] = date[i][:10] + " " + date[i][11:]

    #appending this to the challenge dataframe and save it
    challenge_df = challenge_df.append(to_final(challenge_df,region,"--",list(em_df),list(date)))
    regions_df = regions_df.append(to_final(challenge_df,region,"--",list(em_df),list(date)))

    ####### PRESCRIPTIONS #####################
    # challenge_df.to_csv('data.csv',header=challenge_df.columns,index=False)
    # challenge_df=pd.read_csv("data.csv",
    #                  parse_dates=['Date'],
    #                  encoding="ISO-8859-1",
    #                  dtype={"RegionName": str,
    #                         "RegionCode": str},
    #                  error_bad_lines=True)
    #copying prescriptions from Italy
    #columns = list(challenge_df.columns[5:34]) + list(challenge_df.columns[36:])
    #print("Copying prescriptions from Italy data...")
    # for month in range(1,12):
    #     for day in range(1,32):
    #         date = '2020-'+'{:02d}'.format(month)+'-'+'{:02d}'.format(day)
    #         for column in columns:
    #             index = challenge_df.loc[(challenge_df['CountryName'] == "Italy") & (challenge_df['Date'] == date)].index
    #             for i in range(1,len(index)):
    #                 challenge_df.loc[index[i],column] =  challenge_df.loc[index[0],column]


regions_df["Date"] = pd.to_datetime(regions_df['Date']).dt.date
print("Saving...")
regions_df.to_csv('data/regions.csv',header=regions_df.columns,index=False)
print("Saved data to data/regions.csv")


# CITIES #########################

print("DISTRICTS:")
print("Downloading data from Protezione Civile...")
# REGIONS #######
file = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"
df = pd.read_csv(file)
regions = list(df["denominazione_regione"].unique())
districts = list(df["denominazione_provincia"].unique())
districts_df = pd.DataFrame()
for district in districts:
    print(district + "...                                          ", end="\r")
    total_cases = df[df["denominazione_provincia"] == district]
    em_df = total_cases["totale_casi"].values
    date = total_cases["data"].values
    region = total_cases["denominazione_regione"].values
    for i in range(len(date)):
      date[i] = date[i][:10] + " " + date[i][11:]

    #appending this to the challenge dataframe and save it
    challenge_df = challenge_df.append(to_final(challenge_df,list(region),district,list(em_df),list(date)))
    districts_df = districts_df.append(to_final(challenge_df,list(region),district,list(em_df),list(date)))

    ####### PRESCRIPTIONS #####################
    # challenge_df.to_csv('data.csv',header=challenge_df.columns,index=False)
    # challenge_df=pd.read_csv("data.csv",
    #                  parse_dates=['Date'],
    #                  encoding="ISO-8859-1",
    #                  dtype={"RegionName": str,
    #                         "RegionCode": str},
    #                  error_bad_lines=True)
    #copying prescriptions from Italy
    #columns = list(challenge_df.columns[5:34]) + list(challenge_df.columns[36:])
    #print("Copying prescriptions from Italy data...")
    # for month in range(1,12):
    #     for day in range(1,32):
    #         date = '2020-'+'{:02d}'.format(month)+'-'+'{:02d}'.format(day)
    #         for column in columns:
    #             index = challenge_df.loc[(challenge_df['CountryName'] == "Italy") & (challenge_df['Date'] == date)].index
    #             for i in range(1,len(index)):
    #                 challenge_df.loc[index[i],column] =  challenge_df.loc[index[0],column]

districts_df["Date"] = pd.to_datetime(districts_df['Date']).dt.date
cols = list(districts_df.columns[:-1])
districts_df = districts_df[cols[:3] + ["DistrictName"] + cols[4:]]
#print(" ")
#print("Saving...")
districts_df.to_csv('data/districts.csv',header=districts_df.columns,index=False)
print("Saved data to data/districts.csv")



######################### BOLOGNA DATA ###########################
# print("Formatting Bologna data...")
# # Bologna file
# df = pd.read_excel("casi sintomatici e asintomatici al 13 novembre.xlsx")
#
# # modifyng date format
# df['DATA ACCETTAZIONE'] =pd.to_datetime(df['DATA ACCETTAZIONE'])
# df.sort_values(by='DATA ACCETTAZIONE')
# # now we count the number of positives for each day
# date, daily_cases = np.unique(df["DATA ACCETTAZIONE"], return_counts=True)
#
# # now we sum the daily cases in order to have the absolute number of cases each day (as it is in the challenge file)
# total_cases = []
# total_cases.append(daily_cases[0])
# for i in range(1,len(daily_cases)):
#     total_cases.append(daily_cases[i] + total_cases[i-1])
#

# #appending this to the challenge dataframe
# challenge_df = challenge_df.append(to_final("Bologna",total_cases,date))

# print("Inserting hospital columns...")
# hosp_df = pd.read_csv("ANDAMENTO_UNI.csv")
# hosp_columns = list(hosp_df.columns[2:])
# for column in hosp_columns:
#     challenge_df[column] = np.nan
#
# hosp_df["RegionName"] = "Bologna"
#
# print("Adding Bologna hospital data...")
# hosp_columns
# for month in range(1,12):
#     for day in range(1,32):
#         date = '2020-'+'{:02d}'.format(month)+'-'+'{:02d}'.format(day)
#         for column in hosp_columns:
#             try:
#                 index = challenge_df.loc[(challenge_df['CountryName'] == "Italy")&(challenge_df["RegionName"] == "Bologna") & (challenge_df['Date'] == date)][column].index
#                 challenge_df.loc[index,column] = hosp_df[(hosp_df["DATA"] == date)][column].values[0]
#             except:
#                 pass
