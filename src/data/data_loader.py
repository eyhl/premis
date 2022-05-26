import pandas as pd
import numpy as np
from src.data.preprocessing import load_thickness_time_series
from src.models.paths import PROJECT_ROOT


def load_uplift(timefreq="M"):
    if timefreq=="M":
        df_kuaq = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "KUAQ_filtered.csv")
        df_mik2 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "MIK2_filtered.csv")
    elif timefreq=="W":
        df_kuaq = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "KUAQ_filtered_week.csv")
        df_mik2 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "MIK2_filtered_week.csv")
        df_mik2.loc[0, "Up"] = df_mik2.loc[1, "Up"]
        # There was a nan value in the first index
    else:
        raise Exception(f"No time series with whis {timefreq} interval")

    df_kuaq = df_kuaq[["Time", "Up"]]
    df_kuaq.columns = ["Time", "Up_kuaq"]
    df_kuaq.Time = pd.to_datetime(df_kuaq.Time)

    df_mik2 = df_mik2[["Time", "Up"]]
    df_mik2.columns = ["Time", "Up_mik2"]
    df_mik2.Time = pd.to_datetime(df_kuaq.Time)

    df_height = load_thickness_time_series(
        PROJECT_ROOT / "data" / "raw" / "thickness.mat", timefreq=timefreq
    )
    df = pd.merge(df_height, df_kuaq, how="inner")
    df = pd.merge(df, df_mik2, how="inner")

    return df


def default_em():
    Nr = np.arange(1, 7).tolist()
    radius = [6371e3, 6341e3, 6311e3, 5951e3, 5701e3, 3480e3]
    density = [3.037e3, 3.037e3, 3.438e3, 3.871e3, 4.978e3, 10.750e3]
    rigidity = [0.50605e11, 0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0.0000e11]
    bulk = [5.7437e10, 5.7437e10, 9.9633e10, 1.5352e11, 3.2210e11, 1.1018e12]
    viscosity = [1.0e55, 1.0e21, 1.0e21, 1.0e21, 2.0e21, 0.0e21]

    df_em = pd.DataFrame(
        np.array([Nr, radius, density, rigidity, bulk, viscosity]).T,
        columns=["Nr", "radius", "density", "rigidity", "bulk", "viscosity"],
    )
    return df_em