filename_hlove = "h.dat"

station_name = "KUAQ"
lat_station = 68.58700000
lon_station = -33.05270000

lat_glacier = 68.645056
lon_glacier = -33.029416

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    else:
        cosg = -1

    gamma = np.arccos(cosg)
    return gamma


# Near field
def funcB(gamma, aread, hlove, nlove):
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

    for j in np.arange(1,DevMAX):
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
