import pandas as pd

from src.data.preprocessing import load_thickness_time_series
from src.models.paths import PROJECT_ROOT


def load_uplift():
    df_kuaq = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "KUAQ_filtered.csv")
    df_kuaq = df_kuaq[["Time", "Up"]]
    df_kuaq.columns = ["Time", "Up_kuaq"]
    df_kuaq.Time = pd.to_datetime(df_kuaq.Time)

    df_mik2 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "MIK2_filtered.csv")
    df_mik2 = df_mik2[["Time", "Up"]]
    df_mik2.columns = ["Time", "Up_mik2"]
    df_mik2.Time = pd.to_datetime(df_kuaq.Time)

    df_height = load_thickness_time_series(
        PROJECT_ROOT / "data" / "raw" / "thickness.mat"
    )
    df = pd.merge(df_height, df_kuaq, how="inner")
    df = pd.merge(df, df_mik2, how="inner")

    return df