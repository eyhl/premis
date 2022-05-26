from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import torch
from scipy.optimize import least_squares

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

def load_thickness_time_series(mat_file: str = '../../data/raw/thickness.mat', save: bool = True) -> pd.DataFrame():
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
    
    df_desc = df.groupby(pd.Grouper(level="Time", freq="W"))[
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

def detrend(x):
    if np.isnan(x).sum() != 0:
        nan_index = np.where(np.isnan(x))[0]
        x[nan_index] = x[nan_index + 1]
    
    t = np.arange(len(x))
    G = np.vstack([t**3, t**2, t, np.ones_like(t)]).T
    a, b, c, d = np.linalg.lstsq(G, x, rcond=None)[0]
    trend = a * t**3 + b * t**2 + c * t + d
    x_detrended = x - trend
    return x_detrended

def ffnn_input_vector(df_em):
    df_params = pd.read_csv('../data/processed/ffnn_variable_normalisation_params.csv')

    x = np.hstack(((df_em["radius"].values - df_params["radius_mu"][0]) / df_params["radius_sigma"][0], 
                    (df_em["density"].values - df_params["density_mu"][0]) / df_params["density_sigma"][0],
                    (df_em["rigidity"].values - df_params["rigidity_mu"][0]) / df_params["rigidity_sigma"][0], 
                    (df_em["bulk"].values - df_params["bulk_mu"][0]) / df_params["bulk_sigma"][0], 
                    (df_em["viscosity"].values - df_params["viscosity_mu"][0]) / df_params["viscosity_sigma"][0]))
    x = np.array(x, dtype=np.float32)
    x = np.insert(x, 0, 80)
    x = torch.tensor(x, dtype=torch.float32)
    return x