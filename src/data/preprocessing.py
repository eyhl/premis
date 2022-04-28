from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

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

if __name__ == '__main__':
    _ = volume_to_mass_time_series()