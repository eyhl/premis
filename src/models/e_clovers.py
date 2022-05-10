import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shlex
import subprocess
import re
from pathlib import Path
import contextlib

PROJECT_ROOT = Path.cwd()


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    try:
        prev_cwd = Path.cwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


def write_earth_model(output_name="earth_M3L70V01c.txt"):
    """make earth model file"""
    header = """# radius,    density,   rigidity    bulk         viscosity
# (m)        (kg/m^3)   (Pa)        (Pa)         (Pa.s)\n"""

    Nr = np.arange(1, 6).tolist()
    radius = [6371e3, 6301e3, 5951e3, 5701e3, 3480e3]
    density = [3.037e3, 3.438e3, 3.871e3, 4.978e3, 10.750e3]
    rigidity = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11, 0.0000e11]
    bulk = [5.7437e10, 9.9633e10, 1.5352e11, 3.2210e11, 1.1018e12]
    viscosity = [1.0e55, 1.0e21, 1.0e21, 2.0e21, 0.0e21]

    N_layers = len(Nr)

    fname_earth_model = output_name
    with open(fname_earth_model, mode="w") as f:
        f.write(header)
        for i in range(N_layers):
            # f.write(f" {Nr[i]} {radius[i]}    {density[i]}    {rigidity[i]}  {bulk[i]}    {viscosity[i]}\n")
            f.write(
                f" {Nr[i]} {np.format_float_scientific(radius[i], precision=3, unique=False, exp_digits=1)}   {np.format_float_scientific(density[i], precision=3, unique=False, exp_digits=1)}  {np.format_float_scientific(rigidity[i], precision=4, unique=False, exp_digits=2)}  {np.format_float_scientific(bulk[i], precision=4, unique=False, exp_digits=2)}   {np.format_float_scientific(viscosity[i], precision=1, unique=False, exp_digits=2)}\n"
            )


def write_e_clovers(conf=None):
    """write config into the bash script"""
    if conf is None:
        CONF = {
            "EARTH_FILE": "earth_M3L70V01c.txt",
            "COMPRESS": "1",
            "DEGREE_RANGE": "0 256 1",
            "LABEL_OUTPUT": "Bench_C_256_O_2",
        }

    filename = "e-clovers_3e_bench_TEMPLATE_mod.sh"
    with open(filename, mode="r") as f:
        text = f.read()

    with open("e-clovers_3e_bench_TEMPLATE.sh", mode="w") as f:
        for k, v in CONF.items():
            text = re.sub(r"\${}".format(k), str(v), text)

        f.seek(0)
        f.write(text)
        f.truncate()


def call_e_clovers():
    """Run bash script"""
    command = "./e-clovers_3e_bench_TEMPLATE.sh v3.5.6_Lin64D"
    args = shlex.split(command)
    print(args)
    subprocess.run(args, check=True)


def read_elastic():
    """Read the elastic love numbers as a pandas dataframe"""
    filename = "LLN_Bench_C_256_O_2.dat"
    df = pd.read_csv(filename, skiprows=12, header=None, delim_whitespace=True)
    cols = ["LL", "k", "h", "l"]
    df.columns = cols

    return df


if __name__ == "__main__":
    with working_directory("./e_clovers"):
        # print(Path.cwd())
        write_earth_model()
        write_e_clovers()
        call_e_clovers()

    # print(Path.cwd())
