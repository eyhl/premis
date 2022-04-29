import os
import shlex
import subprocess
import re

# TABOO_DIR = "../src/models/TABOO"
# os.chdir(TABOO_DIR)

# VISCO from bottom to top *1e21
# VM5a (5 layer model top -> bottom) [1.5, 1.25, 0.75, 0.75, 10, 0.0, 0.0, 0.0, 0.0]
# 9L [3.2, 3.2, 3.2, 3.2, 3.2, 1.5, 0.5, 0.5, 0.5]
MAKE_MODEL = {
    "NV": 5,
    "CODE": 0,
    "THICKNESS_LITHOSPHERE": 90.0,
    "CONTROL_THICKNESS": 0,
    "VISCO": [1.5, 1.25, 0.75, 0.75, 10, 0.0, 0.0, 0.0, 0.0],
}


def taboo_task_1(MAKE_MODEL: dict) -> None:
    """_summary_

    Args:
        BASIC_CONF (dict): LMIN/LMAX degree of Love numbers to be evaluated (max is 512)
        Verbose,
        I_loading, forcing is of type loading (1) or tidal (0) eg. ldcs or tLns

        MAKE_MODEL (dict): NV - number of viscosity layers
        CODE - model type see TABOO task_1 file or documentation
        THICKNESS_LITHOSPHERE - thickness of elastic layer
        CONTROL_THICKNESS - control layer 7 - 9 otherwise ignored
        VISCO - list of viscoelastic coeff for each layer given in 1e21 Pa*s from bottom to top (core mantle boundary to surface)
    """
    BASIC_CONF = {"LMIN": 2, "LMAX": 512, "VERBOSE": 0, "I_LOADING": 1}

    NORMALIZED_RESIDUALS = {
        "IH_RES": 1,
        "IL_RES": 0,
        "IK_RES": 0,
    }
    EL_VLUID_VISCEL = {"H_EL": 1, "L_EL": 0, "K_EL": 0}

    HEAVISIDE_TH = {
        "N_HARM_DEG": 2,
        "H_LMIN": 2,
        "H_LMAX": 26,
        "LB_WINDOW": "1e-3",
        "UB_WINDOW": "1e+5",
        "N_POINTS": 11,
        "H_HEAV": 1,
        "L_HEAV": 0,
        "K_HEAV": 0,
    }

    filename = "task_1_mod.dat"
    with open(filename, mode="r") as f:
        text = f.read()

    with open("task_1.dat", mode="w") as f:
        for k, v in BASIC_CONF.items():
            text = re.sub(r"\${}".format(k), str(v), text)

        for k, v in MAKE_MODEL.items():
            if k == "VISCO":
                for idx, item in enumerate(v):
                    text = re.sub(r"\$VISCO_{}".format(idx + 1), str(item), text)
            else:
                text = re.sub(r"\${}".format(k), str(v), text)

        for k, v in NORMALIZED_RESIDUALS.items():
            text = re.sub(r"\${}".format(k), str(v), text)

        for k, v in EL_VLUID_VISCEL.items():
            text = re.sub(r"\${}".format(k), str(v), text)

        for k, v in HEAVISIDE_TH.items():
            text = re.sub(r"\${}".format(k), str(v), text)

        f.seek(0)
        f.write(text)
        f.truncate()

        return


if __name__ == "__main__":
    taboo_task_1(BASIC_CONF, MAKE_MODEL)
    # print("configs written to task_1")

    command = "./TABOO.exe"
    args = shlex.split(command)
    # print(args)
    subprocess.run(args)
