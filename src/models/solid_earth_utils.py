import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import shlex
import os
import src.models

from src.models.paths import PROJECT_ROOT
from src.models.TABOO.taboo import taboo_task_1

filename_hlove = "h.dat"

station_name = "KUAQ"
lat_station = 68.58700000
lon_station = -33.05270000

lat_glacier = 68.645056
lon_glacier = -33.029416


def read_hlove(filename: str):
    h = pd.read_csv(filename, skiprows=2, header=None, delim_whitespace=True)
    n_love = h.shape[0]
    h_love = np.zeros(n_love)
    h_love = h[1].values

    return h_love, n_love


def near_angular_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# ar is load (index 1) and cr is location at station (index 2)
def CompuGamma(t1: float, f1: float, t2: float, f2: float):
    t1 = t1 * np.pi / 180
    t2 = t1 * np.pi / 180
    f1 = t1 * np.pi / 180
    f2 = t1 * np.pi / 180

    cosg = (np.cos(t1) * np.cos(t2)) + (np.sin(t1) * np.cos(t2) * np.cos(f1 - f2))

    if cosg > 1:
        cosg = 1
    elif:
        cosg = -1

    gamma = np.arccos(cosg)
    return gamma


# Near field
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
    coeff = 4 * np.pi * (aa**3) * rho / ma

    x = np.cos(gamma)

    # creo poly l
    polyl = np.zeros(DevMAX + 1)
    p0 = 1
    polyl[0] = p0
    p1 = x
    polyl[1] = p1

    for j in np.arange(1, DevMAX):
        p2 = ((2 * j + 1) * x * p1 - j * p0) / (j + 1)
        polyl[j + 1] = p2
        p0 = p1
        p1 = p2

    x = np.cos(r / aa)

    p0 = 1
    p1 = x
    toth = 0

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


def compute_love_numbers(MAKE_MODEL: dict) -> tuple:
    """wrapper function for generating love numbers based on input earth model

    Args:
        MAKE_MODEL (dict): Input earh model parameters.

    Returns:
        tuple(np.ndarray, np.ndarray): h love numbers and n love numbers respectively
    """
    BASEDIR = os.path.dirname(src.models.__file__)
    print(BASEDIR)
    os.chdir(BASEDIR + "/TABOO")
    taboo_task_1(MAKE_MODEL)

    command = "./taboo.exe"
    args = shlex.split(command)
    # print(args)
    subprocess.run(args)
    os.chdir(BASEDIR)
    filename_hlove = PROJECT_ROOT / "src" / "models" / "TABOO" / "h.dat"
    hlove, nlove = read_hlove(filename_hlove)

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
    print(PROJECT_ROOT)
    MAKE_MODEL = {
        "NV": 5,
        "CODE": 0,
        "THICKNESS_LITHOSPHERE": 90.0,
        "CONTROL_THICKNESS": 0,
        "VISCO": [1.5, 1.25, 0.75, 0.75, 10, 0.0, 0.0, 0.0, 0.0],
    }
    # need to be in taboo folder to run taboo_task_1
    BASEDIR = os.getcwd()
    os.chdir(BASEDIR + "/TABOO")
    taboo_task_1(MAKE_MODEL)

    command = "./TABOO.exe"
    args = shlex.split(command)
    # print(args)
    subprocess.run(args)
    os.chdir(BASEDIR)
    filename_hlove = PROJECT_ROOT / "src" / "models" / "TABOO" / "h.dat"
    hlove, nlove = read_hlove(filename_hlove)

    gamma = CompuGamma(lat_glacier, lon_glacier, lat_station, lon_station)
    near_ang_dist = np.sqrt(
        (lat_glacier - lat_station) ** 2 + (lon_glacier - lon_station) ** 2
    )
    arsurf = 10e5
    ak = 6371e3  # same as aa
    rj = np.sqrt(arsurf / np.pi)
    if gamma >= (3 * rj / ak):
        valure = funcC(gamma, arsurf, hlove, nlove)
    else:
        valore = funcB(gamma, arsurf, hlove, nlove)
        # print(valore)
    print(valore)
