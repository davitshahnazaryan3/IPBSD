import numpy as np
import pandas as pd


def spline(cp_mu, cp_R):
    N = [4, 6, 10, 18, 34]
    mu_d = []
    R_d = []
    for j in range(0, len(N)):
        for ele in cp_mu:
            mu_d.append(ele)
            mu_d.append(ele)
        mu_d_av = np.array(mu_d)

        for i in range(1, len(mu_d)):
            mu_d_av[i] = (mu_d[i] + mu_d[i - 1]) / 2

        mu_d_av2 = np.array(mu_d_av)
        for i in range(1, len(mu_d_av)):
            mu_d_av2[i] = (mu_d_av[i] + mu_d_av[i - 1]) / 2
        cp_mu = mu_d_av2[-N[j]:]
        mu_d = []

        for ele in cp_R:
            R_d.append(ele)
            R_d.append(ele)
        R_d_av = np.array(R_d)

        for i in range(1, len(R_d)):
            R_d_av[i] = (R_d[i] + R_d[i - 1]) / 2
        R_d_av2 = np.array(R_d_av)

        for i in range(1, len(R_d_av)):
            R_d_av2[i] = (R_d_av[i] + R_d_av[i - 1]) / 2
        cp_R = R_d_av2[-N[j]:]
        R_d = []

    # ductility and R factor
    mu = np.exp(cp_mu)
    R = np.exp(cp_R)
    return mu, R


def read_spo_data(filename):
    """
    spo parameters, initial assumption for the definition of the backbone curve
    :param filename: str                            Filename containing spo assumptions as path 'path/*.csv'
    :return: dict                                   Backbone curve parameters
    """
    data = pd.read_csv(filename)
    data = {col: data[col].dropna().to_dict() for col in data}
    spo = {'mc': data["mc"][0], 'a': data["a"][0], 'ac': data["ac"][0], 'r': data["r"][0], 'mf': data["mf"][0],
           'pw': data["pw"][0], 'T': None}
    return spo
