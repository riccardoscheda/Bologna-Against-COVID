import pandas as pd
import numpy as np

# Keep only columns of interest
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


def mov_avg(df, window=7, col="NewCases"):
    """ Returns a new column with the moving average for new cases
    """
    MA = pd.Series(dtype=np.float64)
    for geo in df.GeoID.unique():
        MA = MA.append(df[df["GeoID"] == geo][col].rolling(window=window).mean())
    df["MA"] = MA
    return df


def add_population_data(df):
    """
    Add additional data like population, Cancer rate, etc..  in Oxford data.
    But now it removes rows with at least 1 Nan
    """
    more_df = pd.read_csv("data/Additional_Context_Data_Global.csv")
    more_df.dropna(inplace=True)
    new_df=more_df.merge(df,how='left',left_on=['CountryName',"CountryCode"],right_on=['CountryName',"CountryCode"])
    return new_df

def add_temp(df):
    '''Use this only on the Oxford dataframe.
    Return the same dataframe with a column temperature taken from data/country_temperatures.csv'''

    df_T=pd.read_csv('data/country_temperatures.csv',parse_dates=['Date'])
    df_T=df.merge(df_T,how='left',left_on=['CountryName','Date'],right_on=['CountryName','Date'])
    return df_T


# Helpful function to compute mae
def mae(pred, true):
    return np.mean(np.abs(pred - true))

#  This Function need to be outside the training process


def create_dataset(df):
    """
    From XPRIZE jupyter, this function merges country and region, fills any missing cases
    and fills any missing pis
    """
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)
    # Add new cases column
    # Add new cases column
    df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in npi_cols:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))


    return df

def skl_format(df, moving_average=False, lookback_days=30):
    """
    Takes data and makes a formatting for sklearn
    """
    # Create training data across all countries for predicting one day ahead
    X_cols = cases_col + npi_cols if not moving_average else ["MA"] + npi_cols
    y_col = cases_col if not moving_average else ["MA"]
    print(y_col)
    X_samples = []
    y_samples = []
    geo_ids = df.GeoID.unique()
    for g in geo_ids:
        gdf = df[df.GeoID == g]

        all_case_data = np.array(gdf[cases_col])
        all_npi_data = np.array(gdf[npi_cols])

        # Create one sample for each day where we have enough data
        # Each sample consists of cases and npis for previous lookback_days
        nb_total_days = len(gdf)
        for d in range(lookback_days, nb_total_days - 1):
            X_cases = all_case_data[d-lookback_days:d]

            # Take negative of npis to support positive
            # weight constraint in Lasso.

            ####### NOTICE THIS WAS NEGATIVE AND IN TEST WAS POSITIVE ##########
            X_npis = all_npi_data[d - lookback_days:d]
            ####################################################################

            # Flatten all input data so it fits Lasso input format.
            X_sample = np.concatenate([X_cases.flatten(),
                                       X_npis.flatten()])
            y_sample = all_case_data[d]
            X_samples.append(X_sample)
            y_samples.append(y_sample)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples).flatten()

    return X_samples, y_samples
