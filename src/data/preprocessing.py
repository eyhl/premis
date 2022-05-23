from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np

def volume_to_mass_time_series(mat_file: str = '../../data/raw/volume_time_series.mat', save: bool = True) -> pd.DataFrame():
    """
    Converts a volume time series to mass time series and outputs all as in a dataframe.

    Args:
        mat_file: Path to a .mat file containing volume (kg/m^3) and various times (decimal years).
        save: Optional save the dataframe to csv.

    Returns:
        A dataframe with (time [yr], mass [kg], volume [kg / m^3]) in the columns
    """
    mat = loadmat(mat_file)

    time = mat['time'].squeeze()
    volume = mat['volume'].squeeze()

    # covert to mass
    rho = 917 # density of ice
    mass = rho * volume

    d = {'time': time, 'mass': mass, 'volume': volume}
    df = pd.DataFrame(data=d)

    if save:
        df.to_csv('../../data/processed/mass_time_series.csv')

    return df

def load_thickness_time_series(mat_file: str = '../../data/raw/thickness_time_series.mat', save: bool = True) -> pd.DataFrame():
    """
    Loads a thickness time series from .mat file and preprocesses it to be yearly changes [mm/yr] and 
    resamples to monthly means (placed at last date of month)

    Args:
        mat_file: Path to a .mat file containing volume (kg/m^3) and various times (decimal years).
        save: Optional save the dataframe to csv.

    Returns:
        A dataframe with (time [date], change in thickness [mm/yr]) in the columns
    """
    mat = loadmat(mat_file)

    time = mat['times'].squeeze()
    thickness = mat['thickness'].squeeze()


    d = {'Time': time, 'Thickness': thickness}
    df = pd.DataFrame(data=d)
    #df = df.diff() # running change
    df = df.dropna()
    df = year_convert(df)
    
    df_desc = df.groupby(pd.Grouper(level="Time", freq="M"))[
        ["Thickness"]
    ].describe()
        
    d = {'Time': df_desc.index, 'Thickness': df_desc.iloc[:, 1].values}
    df = pd.DataFrame(data=d)
    df = df.interpolate('pad')
    df.loc[1:, ('Thickness')] = np.diff(df.Thickness.values) * 1e3 # in mm
    df = df.iloc[1:, :]
    #df = df.diff()
    
    if save:
        df.to_csv('../data/processed/thickness_time_series.csv')

    return df


def year_convert(df: pd.DataFrame) -> pd.DataFrame:
    '''Convert dataframe year with xxxx.xx format to datetime yyyy-mm-dd'''
    Year_datetime = pd.to_datetime(df.Time)
    for idx, Year in enumerate(df.Time):
        start = Year
        year = int(start)
        rem = start - year

        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )
        Year_datetime[idx] = result

    Year_datetime = Year_datetime.dt.floor("D")
    df.Time = Year_datetime
    df = df[~df.Time.duplicated()]
    df.index = df.Time
    df = df.drop('Time', axis=1)
    df = df.asfreq('D')

    return df


def year_fraction(df: pd.DataFrame) -> pd.DataFrame:
    # toordinal() returns proleptic Gregorian ordinal of the date,
    # where January 1 of year 1 has ordinal 1
    if isinstance(df.index, pd.DatetimeIndex):
        year_frac = np.zeros(df.shape[0])
        for idx, Date in enumerate(df.index):
            #year_frac[idx] = year_fraction(date)

            start = date(Date.year, 1, 1).toordinal()
            year_length = date(Date.year + 1, 1, 1).toordinal() - start
            year_frac[idx] = Date.year + float(Date.toordinal() - start) / year_length
        df.index = year_frac
        df.dropna(subset=['Up'])
        # df.dropna(axis=0)
    return df