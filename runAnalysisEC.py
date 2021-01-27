from client.master import Master
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from external.getT1 import GetT1


def spectra_par(soil, type_spectra, damping):
    eta = max(np.sqrt(10 / (5 + damping * 100)), 0.55)
    if type_spectra == 1:
        if soil == 'A':
            S = 1.0
            Tb = 0.15
            Tc = 0.4
            Td = 2.
        if soil == 'B':
            S = 1.2
            Tb = 0.15
            Tc = 0.5
            Td = 2.
        if soil == 'C':
            S = 1.15
            Tb = 0.2
            Tc = 0.6
            Td = 2.
        if soil == 'D':
            S = 1.35
            Tb = 0.2
            Tc = 0.8
            Td = 2.
        if soil == 'F':
            S = 1.4
            Tb = 0.15
            Tc = 0.5
            Td = 2.
    if type_spectra == 2:
        if soil == 'A':
            S = 1.0
            Tb = 0.05
            Tc = 0.25
            Td = 1.2
        if soil == 'B':
            S = 1.35
            Tb = 0.05
            Tc = 0.25
            Td = 1.2
        if soil == 'C':
            S = 1.5
            Tb = 0.1
            Tc = 0.25
            Td = 1.2
        if soil == 'D':
            S = 1.8
            Tb = 0.1
            Tc = 0.3
            Td = 1.2
        if soil == 'F':
            S = 1.6
            Tb = 0.05
            Tc = 0.25
            Td = 1.2
    return S, Tb, Tc, Td, eta


def get_ECelastic_spectra(PGA, soil_class, type_spectra=1, damping=0.05):
    # elastic RS EC8 3.2.2.2
    # Type 1 spectra
    global S, Tb, Tc, Td, eta
    S, Tb, Tc, Td, eta = spectra_par(soil_class, type_spectra, damping)
    T = np.linspace(0., 4., 401)
    # Sa in g, Sd in cm
    Sa = np.array([])
    Sd = np.array([])
    for i in range(len(T)):
        if T[i] <= Tb:
            Sa = np.append(Sa, (PGA * S * (1 + T[i] / Tb * (eta * 2.5 - 1))))
        elif Tb < T[i] <= Tc:
            Sa = np.append(Sa, (PGA * S * eta * 2.5))
        elif Tc < T[i] <= Td:
            Sa = np.append(Sa, (PGA * S * eta * 2.5 * Tc / T[i]))
        elif Td < T[i] <= 4:
            Sa = np.append(Sa, (PGA * S * eta * 2.5 * Tc * Td / T[i] ** 2))
        else:
            print('Wrong period range!')
        Sd = np.append(Sd, (Sa[i] * 9.81 * T[i] ** 2 / 4 / np.pi ** 2 * 100))
    return T, Sa


def get_ECdesign_spectra(PGA, soil_class, q, type_spectra=1, damping=0.05, beta=0.2):
    # elastic RS EC8 3.2.2.2
    # Type 1 spectra
    global S, Tb, Tc, Td, eta
    S, Tb, Tc, Td, eta = spectra_par(soil_class, type_spectra, damping)
    T = np.linspace(0., 4., 401)
    # Sa in g, Sd in cm
    Sa = np.array([])
    Sd = np.array([])
    for i in range(len(T)):
        if T[i] <= Tb:
            Sa = np.append(Sa, (PGA * S * (2 / 3 + T[i] / Tb * (2.5 / q - 2 / 3))))
        elif Tb < T[i] <= Tc:
            Sa = np.append(Sa, (PGA * S * 2.5 / q))
        elif Tc < T[i] <= Td:
            Sa = np.append(Sa, (max(beta * PGA, PGA * S * 2.5 / q * Tc / T[i])))
        elif Td < T[i] <= 4:
            Sa = np.append(Sa, (max(beta * PGA, PGA * S * 2.5 / q * Tc * Td / T[i] ** 2)))
        else:
            print('Wrong period range!')
        Sd = np.append(Sd, (Sa[i] * 9.81 * T[i] ** 2 / 4 / np.pi ** 2 * 100))
    return T, Sa


def _hazard(coef, TR, beta_al):
    x = np.linspace(0.005, 3.0, 201)
    k0 = coef['k0']
    k1 = coef['k1']
    k2 = coef['k2']

    # Ground shaking MAFE
    H = [1 / tr for tr in TR]
    p = 1 / (1 + 2 * k2 * (beta_al ** 2))
    Hs = float(k0) * np.exp(-float(k2) * np.log(x) ** 2 - float(k1) * np.log(x))
    MAF = np.sqrt(p) * k0 ** (1 - p) * Hs ** p * np.exp(0.5 * p * k1 ** 2 * (beta_al ** 2))
    p = 1 / (1 + 2 * k2 * (np.power(beta_al, 2)))
    lambdaLS = np.sqrt(p) * k0 ** (1 - p) * H ** p * np.exp(0.5 * p * np.power(k1, 2) * (np.power(beta_al, 2)))
    PGA = np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(lambdaLS / k0))) / 2 / k2)
    return lambdaLS, PGA, MAF, x


def getIndex(x, data):
    if np.where(data >= x)[0].size == 0:
        return np.nan
    else:
        return np.where(data >= x)[0][0]


directory = Path.cwd().parents[0]
outputPath = directory / ".applications/case1/OutputEC"

ipbsd = Master(directory)
input_file = directory / ".applications/case1/ipbsd_input.csv"
hazard_file = directory / ".applications/case1/Hazard-LAquila-Soil-C.pkl"
ipbsd.read_input(input_file, hazard_file, outputPath=outputPath)

# Cross-sections
cs = {'he1': 0.4, 'hi1': 0.45, 'b1': 0.4, 'h1': 0.55,
      'he2': 0.4, 'hi2': 0.45, 'b2': 0.4, 'h2': 0.55,
      'he3': 0.35, 'hi3': 0.4, 'b3': 0.35, 'h3': 0.5,
      'he4': 0.35, 'hi4': 0.4, 'b4': 0.35, 'h4': 0.5,
      'he5': 0.35, 'hi5': 0.4, 'b5': 0.35, 'h5': 0.5, 'T': 0.6}

# Input information
TR = [475]
beta_al = .3
fstiff = .5
runOp = False
ductClass = "DCM"
impClass = "II"

# Global inputs
i_d = ipbsd.data.i_d
q_floor = i_d['bldg_ch'][0]
q_roof = i_d['bldg_ch'][1]
A_floor = i_d['bldg_ch'][2]
heights = np.zeros(len(i_d['h_storeys']))
for storey in range(len(i_d['h_storeys'])):
    heights[storey] = i_d['h_storeys'][storey]
nst = len(heights)

spans_x = []
spans_y = []
for bay in i_d['spans_X']:
    spans_x.append(i_d['spans_X'][bay])
for bay in i_d['spans_Y']:
    spans_y.append(i_d['spans_Y'][bay])

bay_perp = spans_y[0]
n_bays = len(spans_x)
masses = np.zeros(nst)
pdelta_loads = np.zeros(nst)
for storey in range(nst):
    if storey == nst - 1:
        masses[storey] = q_roof * A_floor / 9.81
        pdelta_loads[storey] = q_roof * (sum(spans_y) - bay_perp) * sum(spans_x)
    else:
        masses[storey] = q_floor * A_floor / 9.81
        pdelta_loads[storey] = q_floor * (sum(spans_y) - bay_perp) * sum(spans_x)
fy = i_d['fy'][0]
elastic_modulus_steel = i_d['Es'][0]
eps_y = fy / elastic_modulus_steel
fc = i_d['fc'][0]
n_seismic = i_d['n_seismic_frames'][0]
n_gravity = i_d['n_gravity_frames'][0]
Ec = 3320 * np.sqrt(fc) + 6900
q_beam_floor = bay_perp / 2 * q_floor
q_beam_roof = bay_perp / 2 * q_roof

# Deriving Fb distribution based on assumed T1
Ct = 0.075
T1 = round(Ct * sum(heights) ** (3 / 4), 1)
soil_class = 'C'
type_spectra = 1
damping = 0.05
S, Tb, Tc, Td, eta = spectra_par(soil_class, type_spectra, damping)

# Run MA
b_col = []
b_col_int = []
b_beam = []
h_beam = []

for i in range(nst):
    b_col.append(cs[f"he{i + 1}"])
    b_col_int.append(cs[f"hi{i + 1}"])
    b_beam.append(cs[f"b{i + 1}"])
    h_beam.append(cs[f"h{i + 1}"])

A_cols = []
I_cols = []
A_beams = []
I_beams = []
A_c_ints = []
I_c_ints = []
for i in range(nst):
    A_cols.append(b_col[i] * b_col[i])
    I_cols.append(b_col[i] * b_col[i] ** 3 / 12)
    A_c_ints.append(b_col_int[i] * b_col_int[i])
    I_c_ints.append(b_col_int[i] * b_col_int[i] ** 3 / 12)
for i in range(nst):
    A_beams.append(b_beam[i] * h_beam[i])
    I_beams.append(b_beam[i] * h_beam[i] ** 3 / 12)

gt = GetT1(A_cols, A_c_ints, I_cols, I_c_ints, A_beams, I_beams, nst, spans_x, heights, masses, n_seismic, fc, fstiff,
           just_period=True, w_seismic={'roof': 0, 'floor': 0})
T1, phi_norm = gt.run_ma()

# From MA with corrected stiffnesses
T1 = 1.6

# Hazard
haz_dir = directory / ".applications/case1/OutputEC"
with open(haz_dir / 'coef_Hazard-LAquila-Soil-C.pkl', 'rb') as file:
    coefs = pickle.load(file)
T1_tag = 'SA(%.2f)' % 0.3 if round(T1, 1) == 0.3 else 'SA(%.1f)' % T1

lambdaLS, SaT1, MAF, Sa_haz = _hazard(coefs[T1_tag], TR, beta_al)
lambdaLS, PGA, MAF, Sa_haz = _hazard(coefs['PGA'], TR, beta_al)

# EC8 table 5.1, 5.2.2.2
# assuming DCM and multi-storey multi-bay frame
q0 = 3 * 1.3 if ductClass == 'DCM' else 4.5 * 1.3
# for frame and frame equivalent dual systems
kw = 1
q = max(1.5, q0 * kw)
yI = 0.8 if impClass == 'I' else 1.0 if impClass == 'II' else 1.2 if impClass == 'III' else 1.4
T, Sa = get_ECelastic_spectra(PGA, soil_class)
SaFactor = float(Sa[getIndex(T1, T)] / SaT1)
Sa = Sa / SaFactor / q if T1 >= Tb else (5 / 3 + T1 / Tb * (2.5 / q - 2 / 3)) / (
        1 + T1 / Tb * (2.5 - 1)) * Sa / SaFactor
Lam = 0.85 if (T1 <= 2 * Tc) and (nst > 2) else 1
SaT1 = Sa[getIndex(T1, T)] * yI
m = masses / n_seismic
Fb = SaT1 * 9.81 * sum(m) * Lam
z = np.cumsum(heights)
Fi = np.array([float(Fb * m[i] * z[i] / sum(map(lambda x, y: x * y, z, m))) for i in range(nst)])

# Getting the demands
analysis_type = 3
hinge = None
gravityLoads = [q_beam_floor, q_beam_floor, q_beam_floor, q_beam_floor, q_beam_roof]
demands = ipbsd.run_analysis(analysis_type, cs, Fi, gravityLoads, fstiff=fstiff, hinge=hinge)

# Design the elements
modes = {"Modes": 1 / heights}
details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
    ipbsd.design_elements(demands, cs, modes=modes, dy=None, cover=0.03, est_ductilities=False)

print(Fb, T1, SaT1, Fi)

"""
Inputs necessary for RCMRF
hinge_models.csv
materials.csv
action.csv

"""
import json


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


export_results(outputPath / "hinge_models", hinge_models, "csv")
