from client.master import Master
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from external.getT1 import GetT1
from scipy.stats import lognorm
from scipy.interpolate import interp1d
import os
import scipy
import matplotlib.pyplot as plt


def _spectra(plot_flag, soil, type_spectra, damping, PGA):
    S, Tb, Tc, Td, eta = spectra_par(soil, type_spectra, damping)

    T = np.linspace(0.0, 4., 401)

    for j in range(len(PGA)):
        # Sa in g, Sd in cm
        Sa = []
        Sd = []

        for i in range(len(T)):
            if T[i] <= Tb:
                Sa.append(PGA[j] * S * (1 + T[i] / Tb * (eta * 2.5 - 1)))
            elif Tb < T[i] <= Tc:
                Sa.append(PGA[j] * S * eta * 2.5)
            elif Tc < T[i] <= Td:
                Sa.append(PGA[j] * S * eta * 2.5 * Tc / T[i])
            elif Td < T[i] <= 4:
                Sa.append(PGA[j] * S * eta * 2.5 * Tc * Td / T[i] ** 2)
            else:
                print('Wrong period range!')

            Sd.append(Sa[i] * 9.81 * T[i] ** 2 / 4 / (np.pi) ** 2 * 100)

        if PGA[j] == PGA[0]:
            # SLS case, storing the data
            temp1 = Sd
            temp2 = Sa

            if plot_flag == 'Y':
                # plt.plot(Sd,Sa, label = lbls[j])
                plt.plot(T, Sa)
                plt.ylabel('Spectra Acceleration [g]')
                plt.xlabel('Period [s]')
                plt.grid(True, which="both", ls="--", lw=0.5)
    return Sd, Sa, T


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


def _hazard(coef, beta_al):
    x = np.linspace(0.001, 3.5, 201)
    k0 = coef['k0']
    k1 = coef['k1']
    k2 = coef['k2']

    # Ground shaking MAFE
    p = 1 / (1 + 2 * k2 * (beta_al ** 2))
    Hs = float(k0) * np.exp(-float(k2) * np.log(x) ** 2 - float(k1) * np.log(x))
    MAF = np.sqrt(p) * k0 ** (1 - p) * Hs ** p * np.exp(0.5 * p * k1 ** 2 * (beta_al ** 2))

    return Hs, MAF, x


def getIndex(x, data):
    if np.where(data >= x)[0].size == 0:
        return np.nan
    else:
        return np.where(data >= x)[0][0]


directory = Path.cwd().parents[0]
outputPath = directory / ".applications/case1/OutputZ"

ipbsd = Master(directory)
input_file = directory / ".applications/case1/ipbsd_input.csv"
hazard_file = directory / ".applications/case1/Hazard-LAquila-Soil-C.pkl"
ipbsd.read_input(input_file, hazard_file, outputPath=outputPath)

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

mafc_target = 2.e-4
T1 = 1.2
beta_al = .3
print("Original Period: ", T1)

####################################################################################################
#                               DIRECT FORMULATION                                                 #
####################################################################################################
formulation = 'Direct'

with open(outputPath / 'coef_Hazard-LAquila-Soil-C.pkl', 'rb') as file:
    coefs = pickle.load(file)
T1_tag = 'SA(%.2f)' % 0.3 if round(T1, 1) == 0.3 else 'SA(%.1f)' % T1

Hs, MAF, Sa_haz = _hazard(coefs[T1_tag], beta_al)

# Step 3
# according to Dolsek et al 2017
yls = 1.15
beta_Sec = 0.4


def objective(Say):
    #    global mu_lnR, std_lnR, Sa_step, slopes_new, mafc, new_dist, Sa_new
    # Mean of the lognormal distribution
    mu_lnR = Say

    # Standard deviation of the lognormal distribution
    std_lnR = beta_Sec
    # Getting the lognormal distribution of Sa
    # UPDATE TO USE SA_HAZ INSTEAD?
    Sa = np.linspace(0, 5, 500)
    log_dist = lognorm.cdf(Sa, std_lnR, 0, mu_lnR)

    ## Integration of Sa distribution with the hazard curve
    # Interpolation of Sa lognormal distribution for a common Sa_haz
    interpolator = interp1d(Sa, log_dist)
    new_dist = np.array([])
    Sa_new = np.array([])
    for i in Sa_haz:
        if i >= min(Sa) and i <= max(Sa):
            Sa_new = np.append(Sa_new, i)
            new_dist = np.append(new_dist, interpolator(i))

    # Step of Sa
    Sa_step = Sa_haz[1] - Sa_haz[0]
    slopes = abs(np.gradient(Hs, Sa_step))
    slopes_new = np.array([])
    for i in range(len(Sa_haz)):
        if Sa_haz[i] in Sa_new:
            slopes_new = np.append(slopes_new, slopes[i])
    mafc = np.trapz(y=np.array(new_dist) * np.array(slopes_new), x=np.array(Sa_new))
    return mafc_target - mafc


Satarget = 0.05
Satarget = float(scipy.optimize.fsolve(objective, Satarget))
Se_nc_a = Satarget / yls
print("Satarget", Satarget)

# step 4 - SDOF model with a T1=T1, and ductility=muNC, validate at a Sa close to Se_nc_a
rs = 2
muNC = 6

M = np.zeros([nst, nst])
n_seismic = i_d['n_seismic_frames'][0]
m = np.zeros(nst)
for i in range(nst):
    if i == nst - 1:
        m[i] = q_roof * A_floor / 9.81 / n_seismic
    else:
        m[i] = q_floor * A_floor / 9.81 / n_seismic

H = np.array([[sum(heights[0:i + 1])] for i in range(nst)])
if nst <= 4:
    phi = np.array([H[i] / H[nst - 1] for i in range(nst)])
else:
    phi = np.array([4 / 3 * H[i] / H[nst - 1] * (1 - H[i] / 4 / H[nst - 1]) for i in range(nst)])
for i in range(nst):
    M[i][i] = m[i]
identity = np.ones((1, nst))
Me = float(
    np.matmul(np.matmul(phi.transpose(), M), identity.transpose()) ** 2 / np.matmul(np.matmul(phi.transpose(), M), phi))
He = float(sum(map(lambda x, y, z: x * y * z, sum(M), np.array([H[i] / H[nst - 1] for i in range(nst)]), \
                   np.array([sum(heights[0:i + 1]) for i in range(nst)]))) / \
           sum(map(lambda x, y: x * y, sum(M), np.array([H[i] / H[nst - 1] for i in range(nst)]))))
Te = 2.4
Ke = 4 * np.pi ** 2 * Me / Te ** 2
EIe = Ke * (He ** 3) / 3
fc = 25
Ec = (3320 * np.sqrt(fc) + 6900) * 1000
Ie = EIe / Ec
be = (Ie * 12) ** (1 / 4)

Mt = sum(sum(M))
Kg = 4 * np.pi ** 2 * Me / T1 ** 2
EIg = Kg * (He ** 3) / 3
Ig = EIg / Ec
b = (Ig * 12) ** (1 / 4)
Mye = .25 * Se_nc_a * He * Me * 9.81
dy_sdof = .25 * Se_nc_a * 9.81 * (T1 / 2 / np.pi) ** 2

# print(Me, He, Mye, dy_sdof, b)


def text_read(name, col):
    f = open(name, 'r')
    lines = f.readlines()
    data = np.array([])
    for x in lines:
        data = np.append(data, float(x.split()[col]))
    f.close()
    return data


def checkMy(My, data):
    if np.where(data >= My)[0].size == 0:
        return np.nan
    else:
        return np.where(data >= My)[0][0]


periods = text_read(outputPath / formulation / 'Periods.txt', 0)
push_dir = outputPath / formulation / 'SPO'
topDispSdof = text_read(push_dir / 'DFree.txt', 0)
base_shearSdof = np.zeros(len(topDispSdof))
for file in os.listdir(push_dir):
    if file.startswith("base_shear"):
        base_shearSdof = base_shearSdof - text_read(push_dir / file, 0)

maxVloc = checkMy(max(base_shearSdof), base_shearSdof)
for i in range(len(base_shearSdof)):
    if (base_shearSdof[i + 1] - base_shearSdof[i]) / (topDispSdof[i + 1] - topDispSdof[i]) < 0.2 * base_shearSdof[i] / \
            topDispSdof[i]:
        Dy = topDispSdof[i]
        Vy = base_shearSdof[i]
        break

# print(Dy, Vy)

im_qtile_name = outputPath / formulation / 'SDOF' / 'im_qtile.npy'
mtdisp_range_name = outputPath / formulation / 'SDOF' / 'mtdisp_range.npy'
im_qtile = np.load(im_qtile_name, allow_pickle=True)
imtd_med = im_qtile[1]
mtdisp_range = np.load(mtdisp_range_name, allow_pickle=True)
plt.plot(mtdisp_range, imtd_med)

Du = Dy * muNC
omega = 2 * np.pi / periods[0]
indSe = checkMy(Du, mtdisp_range)
Dn_nc = mtdisp_range[indSe]
Se_nc = imtd_med[indSe]
De_nc = Se_nc * 9.81 / omega ** 2
C1 = Dn_nc / De_nc
rNC = muNC / C1 * rs
rmuNC = muNC / C1
print(C1, rs, rNC, muNC, rmuNC)

# Step 5
Se_D_a = Se_nc_a / rNC
# Step 6
# Design spectrum by normalizing EC8 elastic spectrum to Se_D_a
S, Tb, Tc, Td, eta = spectra_par('C', 1, 0.05)

if T1 <= Tb:
    ag = np.array([Se_D_a / S / (1 + T1 / Tb * (eta * 2.5 - 1))])
elif Tb < T1 <= Tc:
    ag = np.array([Se_D_a / S / eta / 2.5])
elif Tc < T1 <= Td:
    ag = np.array([Se_D_a / S / eta / 2.5 / Tc * T1])
elif Td < T1 <= 4:
    ag = np.array([Se_D_a / S / eta / 2.5 / Tc / Td * T1 ** 2])
else:
    raise ValueError('Wrong fundamental period')

Sd, Sa, T = _spectra('N', 'C', 1, 0.05, ag)
# print("Ground acceleration: ", ag * 9.81)

####################################################################################################
#                               INDIRECT FORMULATION                                                 #
####################################################################################################

# Step 1, same as for direct
# Step 2
k0 = coefs[T1_tag]['k0']
k1 = coefs[T1_tag]['k1']
k2 = coefs[T1_tag]['k2']
TR = np.array([475, 10000])
H = np.array([1 / tr for tr in TR])
beta_al = np.array([.2, .3])
p = 1 / (1 + 2 * k2 * (beta_al ** 2))
# Probably Sa(T1) should be computed from a design spectra instead
lambdaLS = np.sqrt(p) * k0 ** (1 - p) * H ** p * np.exp(0.5 * p * np.power(k1, 2) * (np.power(beta_al, 2)))
SaT1 = np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(H / k0))) / 2 / k2)
k = abs((np.log(H[0]) - np.log(H[1])) / (np.log(SaT1[0]) - np.log(SaT1[1])))
# Step 3a
if T1 > 3 * Tc:
    yim_a = 5.0
else:
    yim_a = np.interp(T1, np.array([0, 3 * Tc]), np.array([2.5, 5.0]))
# Step 3b
yim_b = 1 / yls * np.power((TR[0] * mafc_target), (-1 / k)) * np.exp(k / 2 * np.power(beta_Sec, 2))
# Step 4, same as for direct
qa = rNC / yim_b
# Step 5
Se_D_a_indirect = SaT1[0] / qa
# Step 6
ag_ind = np.array([Se_D_a_indirect / S / eta / 2.5 / Tc * T1])
Sd_ind, Sa_ind, T_ind = _spectra('N', 'A', 1, 0.05, ag_ind)

print('A) Risk-targeted design spectral acceleration from the DIRECT formulation is {}g'.format(round(Se_D_a, 3)))
print('Reference peak ground acceleration (Direct) to be used {}g, or {}m/s2'.format(round(float(ag), 3),
                                                                                     round(ag[0] * 9.81, 3)))
print('B) Risk-targeted design spectral acceleration from the INDIRECT formulation is {}g'.format(
    round(Se_D_a_indirect, 3)))
print('Reference peak ground acceleration (Indirect) to be used {}g, or {}m/s2'.format(round(float(ag_ind), 3),
                                                                                       round(ag_ind[0] * 9.81, 3)))

#####
# Cross-sections
cs = {'he1': 0.45, 'hi1': 0.5, 'b1': 0.44, 'h1': 0.6,
      'he2': 0.45, 'hi2': 0.5, 'b2': 0.44, 'h2': 0.6,
      'he3': 0.4, 'hi3': 0.45, 'b3': 0.4, 'h3': 0.55,
      'he4': 0.4, 'hi4': 0.45, 'b4': 0.4, 'h4': 0.55,
      'he5': 0.35, 'hi5': 0.4, 'b5': 0.35, 'h5': 0.55, 'T': 0.6}

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

fstiff = 0.5
gt = GetT1(A_cols, A_c_ints, I_cols, I_c_ints, A_beams, I_beams, nst, spans_x, heights, masses, n_seismic, fc, fstiff,
           just_period=True, w_seismic={'roof': 0, 'floor': 0}, single_mode=False)
T1, phi_norm = gt.run_ma()
identity = np.ones((1, nst))
Gamma = np.zeros(nst)
Lstar = np.zeros(nst)
M1eff = np.zeros(nst)
phi_tran = phi_norm.transpose()
for i in range(nst):
    phi = phi_tran[0:nst, i:i + 1]
    # First mode participation factor
    Gamma[i] = np.matmul(np.matmul(phi.transpose(), M), identity.transpose()) / np.matmul(np.matmul(phi.transpose(), M),
                                                                                          phi)
    # Generalized 1st mode mass
    Lstar[i] = np.matmul(np.matmul(phi.transpose(), M), identity.transpose())
    # 1st mode effective modal mass
    M1eff[i] = np.matmul(np.matmul(phi.transpose(), M), identity.transpose()) ** 2 / np.matmul(
        np.matmul(phi.transpose(), M), phi)

print("Vy", 2*9.81*Se_D_a*Gamma[0]*Lstar[0])

print("Period: ", T1)
ag = max(ag_ind, ag)

T1[0] = 1.2

# Response spectrum
Sd, Sa, T = _spectra('N', 'C', 1, 0.05, ag)
T = np.around(T, decimals=2)
Se = np.zeros(nst)
for i in range(nst):
    Se[i] = Sa[np.where(T == round(T1[i], 2))[0][0]]

print("Se: ", Se)

# %% rows=number of stories
# columns=number of modes
# design seismic forces
sForces = np.zeros([nst, nst])
# equivalent static forces
fStatic = np.zeros([nst, nst])
# design storey shear diagram
V = np.zeros([nst, nst])
gravityLoads = [q_beam_floor, q_beam_floor, q_beam_floor, q_beam_floor, q_beam_roof]
for i in range(nst):
    phi = phi_tran[0:nst, i:i + 1]
    sForces[0:nst, i:i + 1] = Gamma[i] * np.matmul(M, phi)
    fStatic[0:nst, i:i + 1] = sForces[0:nst, i:i + 1] * Se[i] * 9.81

# Use fStatic to apply ELFM via OpenSees for the required number of modes
# Then perform CQC
damp = .05
corr = np.zeros([nst, nst])
for n in range(nst):
    for i in range(nst):
        corr[i, n] = (8 * damp ** 2 * (T1[n] / T1[i]) ** (3 / 2)) / (
                    (1 + (T1[n] / T1[i])) * ((1 - (T1[n] / T1[i])) ** 2 + 4 * damp ** 2 * (T1[n] / T1[i])))

analysis_type = 5
hinge = None
gravityLoads = [q_beam_floor, q_beam_floor, q_beam_floor, q_beam_floor, q_beam_roof]

demands = {}
num_modes = 5
for mode in range(num_modes):
    demands[f"Mode{mode + 1}"] = ipbsd.run_analysis(analysis_type, cs,
                                                    fStatic[:, mode],
                                                    fstiff=fstiff, hinge=hinge)

demands = ipbsd.perform_cqc(corr, demands)

demands_gravity = ipbsd.run_analysis(analysis_type, cs,
                                     grav_loads=gravityLoads,
                                     fstiff=fstiff, hinge=hinge)

# Combining gravity and RSMA results
for eleType in demands_gravity.keys():
    for dem in demands_gravity[eleType].keys():
        if eleType == "Beams" and dem == "M":
            demands[eleType][dem]["Pos"] = demands[eleType][dem]["Pos"] + demands_gravity[eleType][dem]["Pos"]
            demands[eleType][dem]["Neg"] = demands[eleType][dem]["Neg"] + demands_gravity[eleType][dem]["Neg"]
        else:
            demands[eleType][dem] = demands[eleType][dem] + demands_gravity[eleType][dem]

# Design the elements
modes = {"Modes": 1 / heights}
details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
    ipbsd.design_elements(demands, cs, modes=modes, dy=None, cover=0.03, est_ductilities=False)

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
