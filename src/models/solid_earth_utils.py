import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import shlex
import os
import src.models
from numba import jit

from src.models.paths import PROJECT_ROOT

# from src.models.TABOO.taboo import taboo_task_1
from src.models.e_clovers import (
    working_directory,
    write_earth_model,
    write_e_clovers,
    call_e_clovers,
    read_elastic,
)

filename_hlove = "h.dat"

station_name = "KUAQ"
lat_station = 68.58700000
lon_station = -33.05270000

lat_glacier = 68.645056
lon_glacier = -33.029416


def read_hlove(filename: str):
    """jep"""
    h = pd.read_csv(filename, skiprows=2, header=None, delim_whitespace=True)
    n_love = h.shape[0]
    h_love = np.zeros(n_love)
    h_love = h[1].values

    return h_love, n_love


# ar is load (index 1) and cr is location at station (index 2)
def CompuGamma(t1: float, f1: float, t2: float, f2: float):
    """compute angular distance between two sets of coordinates"""
    t1 = t1 * np.pi / 180
    t2 = t2 * np.pi / 180
    f1 = f1 * np.pi / 180
    f2 = f2 * np.pi / 180

    cosg = (np.cos(t1) * np.cos(t2)) + (np.sin(t1) * np.cos(t2) * np.cos(f1 - f2))

    if cosg > 1:
        cosg = 1
    elif cosg < -1:
        cosg = -1

    gamma = np.arccos(cosg)
    return gamma


# Near field
@jit(nopython=True)
def funcB(gamma, aread, hlove, nlove):
    """helo"""
    aa = 6371e3  # Earth Radius in m
    ad = 26000e6  # ?
    rho = 1e3  # density of water or ice?
    ma = 5.9e24  # mass of earth
    DevMAX = 80000  # ?

    hmax = hlove[nlove - 1]  # be sure hlove should not be 1 longer eg. 0 at index 0
    zfac = np.pi / 180
    r = np.sqrt(aread / np.pi)

    # if use height instead of mass
    coeff = 4 * np.pi * (aa**3) * rho / ma
    # if use mass instead of height
    # coeff = aa / ma

    x = np.cos(gamma)

    # creo poly l
    polyl = np.zeros(DevMAX + 1)
    p0 = 1
    polyl[0] = p0
    p1 = x
    polyl[1] = p1

    # legendre polynomial evaluated at arcdistance gamma
    for j in np.arange(1, DevMAX):
        p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
        polyl[j + 1] = p2
        p0 = p1
        p1 = p2

    # disk distance
    x = np.cos(r / aa)

    p0 = 1
    p1 = x
    toth = 0
    # differentiated legandre polynomial evaluated for disk
    # see Valentina paper 2006 eq (4)
    for j in np.arange(1, nlove):
        p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
        pp = ((x * p1) - p2) / j
        elemh = hlove[j] * pp * polyl[j]
        toth = toth + elemh
        p0 = p1
        p1 = p2

    for j in range(nlove + 1, DevMAX):
        p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
        pp = (((x * p1) - p2)) / j
        elemh = hmax * pp * polyl[j]
        toth = toth + elemh
        p0 = p1
        p1 = p2

    valore = (toth / 2) * coeff * 1e3

    return valore


# Far field
def funcC(gamma, aread, hlove, nlove):
    raise NotImplementedError()


def compute_love_numbers(
    df_em: pd.DataFrame = None, CONF: dict = None, verbose: int = 1
):

    """wrapper function for generating love numbers based on input earth model

    Args:
        MAKE_MODEL (dict): Input earh model parameters.

    Returns:
        tuple(np.ndarray, np.ndarray): h love numbers and n love numbers respectively

    """
    if CONF is not None:
        lovefile = f'LLN_{CONF["LABEL_OUTPUT"]}.dat'
    else:
        lovefile = "LLN_Bench_C_256_O_2.dat"

    with working_directory(PROJECT_ROOT / "src" / "models" / "e_clovers"):
        write_earth_model(df_em)
        write_e_clovers(CONF)
        call_e_clovers(verbose=verbose)
        df = read_elastic(filename=lovefile)

    hlove = df.h.values
    nlove = hlove.shape[0]

    return (hlove, nlove)


def greens_function(
    hlove: np.ndarray,
    nlove: np.ndarray,
    glacier_coordinates: list,
    station_coordinates: list,
    arsurf: float = 10e5,
) -> float:
    """_summary_

    Args:
        hlove (np.ndarray): h love numbers
        nlove (np.ndarray): n love numbers
        glacier_coordinates (list): coordinates of the estimated glacier center of mass
        station_coordinates (list): coordinates of the gnss station
        arsurf (float, optional): MISSING DESCRIPTION. Defaults to 10e5.

    Returns:
        float: greens function weight for computing uplift based on mass change
    """

    lat_glacier, lon_glacier = glacier_coordinates
    lat_station, lon_station = station_coordinates
    gamma = CompuGamma(lat_glacier, lon_glacier, lat_station, lon_station)
    gf = funcB(gamma, arsurf, hlove, nlove)
    return gf


if __name__ == "__main__":
    station_coordinates = [68.58700000, -33.05270000]
    glacier_coordinates = [68.645056, -33.029416]
    hlove, nlove = compute_love_numbers()
    gf = greens_function(
        hlove, nlove, glacier_coordinates, station_coordinates, arsurf=5e3 * 5e3
    )
    print(gf)

    """
    with working_directory("./e_clovers"):
        # print(Path.cwd())
        write_earth_model()
        write_e_clovers()
        call_e_clovers()
        df = read_elastic()
    """
