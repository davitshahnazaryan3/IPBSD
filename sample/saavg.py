import os
import numpy as np
from pathlib import Path
import cmath
import pandas as pd
import pickle
from scipy.interpolate import interp1d
from scipy.stats import gmean
from scipy.stats import lognorm

import warnings
warnings.filterwarnings('ignore')


def _responseSpectrum(acc, dt, period, damping):
    if period == 0.0:
        period = 1e-20
    PGA = max(acc)
    pow = 1
    while 2 ** pow < len(acc):
        pow = pow + 1
    nPts = 2 ** pow
    fas = np.fft.fft(acc, nPts)
    dFreq = 1/(dt*(nPts - 1))
    freq = dFreq * np.array(range(nPts))
    if nPts%2 != 0:
        symIdx = int(np.ceil(nPts/2))
    else:
        symIdx = int(1 + nPts/2)
    natFreq = 1/period
    H = np.ones(len(fas), 'complex')
    H[np.int_(np.arange(1, symIdx))] = np.array([natFreq**2 * 1/((natFreq**2 -\
    i**2) + 2*cmath.sqrt(-1) * damping * i * natFreq) for i in freq[1:symIdx]])
    if nPts%2 != 0:
        H[np.int_(np.arange(len(H)-symIdx+1, len(H)))] = \
        np.flipud(np.conj(H[np.int_(np.arange(1, symIdx))]))
    else:
        H[np.int_(np.arange(len(H)-symIdx+2, len(H)))] = \
        np.flipud(np.conj(H[np.int_(np.arange(1, symIdx-1))]))
    sa = max(abs(np.real(np.fft.ifft(np.multiply(H, fas)))))
    return {'PGA':PGA, 'sa':sa}


def text_read(name, col):
    f = open(name, 'r')
    lines = f.readlines()
    data = []
    for x in lines:
        data.append(float(x.split()[col]))
    f.close()
    data = np.array(data)
    return data


def mafe_direct_im_based(H, s, eta, beta):
    """
    Details:
    Compute the MAFE of a limit state defined via a fitted lognormal
    distribution by integrating directly with the  hazard curve
    Treat the hazard input to avoid errors.
    We strip out:
     1. the negative H values (usually at the beginning)dy_model
     2. the points with constant s (usually at the end)
    Information:
    Author: Gerard J. O'Reilly
    First Version: April 2020
    Notes:
    References:
    Porter KA, Beck JL, Shaikhutdinov R V. Simplified Estimation of Economic
    Seismic Risk for Buildings. Earthquake Spectra 2004; 20(4):
    1239–1263. DOI: 10.1193/1.1809129.
    Inputs:
    H: List of values of annual probability of exceedance
    s: List of corresponding intensities for H
    eta: Fragility function median (intensity)
    beta: Fragility function dispersion (total)
    Returns:
    l: Mean annual frequency of exceedance
    """
    from scipy import stats
    import numpy as np

    # Do first strip
    s_f = []
    H_f = []
    for aa, bb in zip(s, H):
        if bb > 0:
            s_f.append(aa)
            H_f.append(bb)

    # Do second strip
    s_ff = []
    H_ff = []
    for i in range(len(s_f) - 1):
        if H_f[i] - H[i + 1] > 0:
            s_ff.append(s_f[i])
            H_ff.append(H_f[i])
    s_ff.append(s_f[-1])
    H_ff.append(H_f[-1])

    # Overwrite the initial variable for convenience
    s = s_ff
    H = H_ff

    # First we compute the PDF value of the fragility at each of the discrete
    # hazard curve points
    p = stats.lognorm.cdf(s, beta, scale=eta)

    # This function computes the MAF using Method 1 outlined in
    # Porter et al. [2004]
    # This assumes that the hazard curve is linear in logspace between
    # discrete points among others

    # Initialise some arrays
    ds = []
    ms = []
    dHds = []
    dp = []
    dl = []

    for i in np.arange(len(s)-1):
        ds.append(s[i+1]-s[i])
        ms.append(s[i]+ds[i]*0.5)
        dHds.append(np.log(H[i+1]/H[i])/ds[i])
        dp.append(p[i+1]-p[i])
        dl.append(p[i]*H[i]*(1-np.exp(dHds[i]*ds[i]))-dp[i]/ds[i]*H[i]*(np.exp(dHds[i]*ds[i])*(ds[i]-1/dHds[i])+1/dHds[i]))

    # Compute the MAFE
    l = sum(dl)

    # Return the MAFE
    return l


directory = Path(os.getcwd())
gm_path = directory.parents[0] / '.applications/case1/GroundMotions'
rs_path = directory.parents[0] / '.applications/case1/Output1/RS.pickle'
outputDir = directory.parents[0] / ".applications/case1/Output1"

with open(outputDir/"Cache/modelOutputs.pickle", 'rb') as file:
    spo = pickle.load(file)

with open(outputDir/"RCMRF/ida_cache.pickle", 'rb') as file:
    ida = pickle.load(file)

dt_file = np.array(pd.read_csv(gm_path/'GMR_dts.txt',header=None)[0])
gm_file = list(pd.read_csv(gm_path/'GMR_names1.txt',header=None)[0])

dr_model = spo["SPO_idealized"][0][-1]
mtdisp = ida["mtdisp"]
im_spl = ida["im_spl"]

spl_interp = interp1d(mtdisp, im_spl)
spl_mu = spl_interp(dr_model)

with open(rs_path, 'rb') as file:
    rs = pickle.load(file)

T1 = 1.00
sat1_list = np.array([])
scaling_factors = np.array([])
idx_t1 = int(T1 * 100)
for i in range(len(gm_file)):
    SaT1 = float(rs[i+1][idx_t1])
    sat1_list = np.append(sat1_list, SaT1)
    scaling_factors = np.append(scaling_factors, spl_mu[i] / SaT1)

Tbot = T1*0.2
Tup = T1*1.5
c_range = np.arange(Tbot, Tup, 0.1)
sa_avgs = np.array([])
for i in range(len(gm_file)):
    rs[i+1] = rs[i+1]*scaling_factors[i]
    spectrum = np.zeros(len(c_range))
    for j in range(len(c_range)):
        idx = int(c_range[j]*100)
        spectrum[j] = float(rs[i+1][idx])
    sa_avgs = np.append(sa_avgs, gmean(spectrum))

eta = np.median(sa_avgs)
beta = np.std(np.log(sa_avgs))

print(sa_avgs, eta, beta)

haz_dir = directory.parents[0]/'.applications'/'case1'
with open(haz_dir/'Hazard-LAquila-Soil-Csaavg.pkl', 'rb') as file:
    [im, s, apoe] = pickle.load(file)
    im = np.array(im)
    s = np.array(s)
    apoe = np.array(apoe)

indx_T = int(T1*10)
mafc = mafe_direct_im_based(apoe[indx_T], s[indx_T], eta, beta)
print(mafc)



