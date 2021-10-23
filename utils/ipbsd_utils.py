"""
Utility functions for IPBSD
"""
import timeit
import os
import numpy as np
import pickle
import json
from numpy import ones, vstack
from numpy.linalg import lstsq
import pandas as pd
from colorama import Fore


def get_init_time():
    """
    Records initial time
    :return: float                      Initial time
    """
    start_time = timeit.default_timer()
    return start_time


def truncate(n, decimals=0):
    """
    Truncates time with given decimal points
    :param n: float                     Time
    :param decimals: int                Decimal points
    :return: float                      Truncated time
    """
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def get_time(start_time):
    """
    Prints running time in seconds and minutes
    :param start_time: float            Initial time
    :return: None
    """
    elapsed = timeit.default_timer() - start_time
    print('Running time: ', truncate(elapsed, 1), ' seconds')
    print('Running time: ', truncate(elapsed / float(60), 2), ' minutes')


def initiate_msg(text):
    print(Fore.LIGHTBLUE_EX + text + Fore.WHITE)


def success_msg(text):
    print(Fore.LIGHTGREEN_EX + text + Fore.WHITE)


def error_msg(text):
    print(Fore.LIGHTRED_EX + text + Fore.WHITE)


def create_folder(directory):
    """
    creates directory
    :param directory: str                   Directory to be created
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def create_and_export_cache(filename, filetype, **kwargs):
    data = {}
    for arg in kwargs:
        data[arg] = kwargs.get(arg, None)

    export_results(filename, data, filetype)


def export_results(filepath, data, filetype):
    """
    Store results in the database
    :param filepath: str                            Filepath, e.g. "directory/name"
    :param data:                                    Data to be stored
    :param filetype: str                            Filetype, e.g. npy, json, pkl, csv
    :return: None
    """
    if filetype == "npy":
        np.save(f"{filepath}.npy", data)
    elif filetype == "pkl" or filetype == "pickle":
        with open(f"{filepath}.pickle", 'wb') as handle:
            pickle.dump(data, handle)
    elif filetype == "json":
        with open(f"{filepath}.json", "w") as json_file:
            json.dump(data, json_file)
    elif filetype == "csv":
        data.to_csv(f"{filepath}.csv", index=False)


def geo_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.mean())


def getIndex(target, data, tol=0.):
    if np.where(data >= target)[0].size == 0:
        return np.nan
    else:
        return np.where(data >= target - tol)[0][0]


def getEquation(p1, p2):
    points = [p1, p2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    return m, c


def compare_value(x, y, tol=0.05):
    """
    Verify whether x is within tolerance bounds of y
    :param x: float
    :param y: float
    :param tol: float
    :return: bool
    """
    if max(x, y) - tol * max(x, y) <= min(x, y) <= max(x, y) + tol * max(x, y):
        return True
    else:
        return False


def compare_areas(x, y, tol=0.2):
    """

    :param x: dict
    :param y: dict
    :param tol: float
    :return: bool
    """
    peak_x = x["a"] * (x["mc"] - 1) + 1
    peak_y = y["a"] * (y["mc"] - 1) + 1

    a1x = (1 + peak_x) / 2 * (x["mc"] - 1)
    a1y = (1 + peak_y) / 2 * (y["mc"] - 1)

    a2x = (x["mf"] - x["mc"]) * peak_x / 2
    a2y = (y["mf"] - y["mc"]) * peak_y / 2

    return compare_value(a1x, a1y, tol=tol) and compare_value(a2x, a2y, tol=tol)


def export(data, fstiff, path, flag3d):
    """
    export cache to path
    :param data: dict                               IPBSD inputs
    :param fstiff: float                            Stiffness reduction factor
    :param path: str                                Path to directory to export the files
    :param flag3d: bool                             3D or 2D
    :return: None
    """
    # Number of storeys
    nst = len(data.data["h_storeys"])

    # Loads (the only parameter needed for 3D)
    q_floor = float(data.data["bldg_ch"][0])
    q_roof = float(data.data["bldg_ch"][1])

    # For 2D only
    spansY = np.array([data.data["spans_Y"][x] for x in data.data["spans_Y"]])
    distLength = spansY[0] / 2

    # Distributed loads
    distLoads = [q_floor * distLength, q_roof * distLength]

    # Point loads, for now will be left as zero
    pLoads = 0.0

    # Number of gravity frames
    if not flag3d:
        nGravity = len(spansY) - 1
    else:
        nGravity = 0

    # PDelta loads/ essentially loads going to the gravity frames (for 2D only)
    if nGravity > 0:
        pDeltaLoad = data.pdelta_loads
    else:
        pDeltaLoad = 0.0

    # Masses (actual frame mass is exported) (for 2D only)
    masses = np.array(data.masses)

    # Creating a DataFrame for loads
    loads = pd.DataFrame(columns=["Storey", "Pattern", "Load"])

    for st in range(1, nst + 1):

        load = distLoads[1] if st == nst else distLoads[0]

        loads = loads.append({"Storey": st,
                              "Pattern": "distributed",
                              "Load": load}, ignore_index=True)

        # Point loads will be left as zeros for now
        loads = loads.append({"Storey": st,
                              "Pattern": "point internal",
                              "Load": pLoads}, ignore_index=True)
        loads = loads.append({"Storey": st,
                              "Pattern": "point analysis",
                              "Load": pLoads}, ignore_index=True)

        # PDelta loads (for 2D only)
        if nGravity > 0:
            # Associated with each seismic frame
            load = pDeltaLoad[st - 1] / data.n_seismic
            loads = loads.append({"Storey": st,
                                  "Pattern": "pdelta",
                                  "Load": load}, ignore_index=True)

        else:
            # Add loads as zero
            loads = loads.append({"Storey": st,
                                  "Pattern": "pdelta",
                                  "Load": pDeltaLoad}, ignore_index=True)

        # Masses (for 2D only)
        loads = loads.append({"Storey": st,
                              "Pattern": "mass",
                              "Load": masses[st - 1] / data.n_seismic}, ignore_index=True)

        # Area loads (for both 2D and 3D)
        q = q_roof if st == nst else q_floor
        loads = loads.append({"Storey": st,
                              "Pattern": "q",
                              "Load": q}, ignore_index=True)

    # Exporting action for use by a Modeler module
    """
    For a two-way slab assumption, load distribution will not be uniform.
    For now and for simplicity, total load over each directions is computed and then divided by global length to 
    assume a uniform distribution. """
    export_results(path / "action", loads, "csv")

    # Materials
    fc = data.data["fc"][0]
    fy = data.data["fy"][0]
    Es = data.data["Es"][0]

    # Elastic modulus of uncracked concrete
    Ec = (3320 * np.sqrt(fc) + 6900) * fstiff

    materials = pd.DataFrame({"fc": fc,
                              "fy": fy,
                              "Es": Es,
                              "Ec": Ec}, index=[0])

    # Exporting the materials file for use by a Modeler module
    export_results(path / "materials", materials, "csv")


def check_for_file(filepath):
    # check whether solutions file was provided
    if not isinstance(filepath, int):
        # If the solution file is not an integer, referring to a row in the csv file
        if filepath is not None:
            # if there is a file provided
            try:
                # csv
                solution = pd.read_csv(filepath)
            except:
                # wrong file type
                solution = None
        else:
            # no filepath or integer were supplied
            solution = None
    else:
        # Read according to row index
        solution = filepath
    return solution
