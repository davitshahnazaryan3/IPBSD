import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# todo, clean this file, make it an object with methods
def spline(cp_mu, cp_R):
    N = [4, 6, 10, 18, 34]
    mu_d = []
    R_d = []
    for j in range(0, len(N)):
        for ele in cp_mu:
            mu_d.append(ele)
            mu_d.append(ele)
        mu_d_av = np.array(mu_d)
        for i in range(1, len(mu_d)): mu_d_av[i] = (mu_d[i] + mu_d[i - 1]) / 2
        mu_d_av2 = np.array(mu_d_av)
        for i in range(1, len(mu_d_av)): mu_d_av2[i] = (mu_d_av[i] + mu_d_av[i - 1]) / 2
        cp_mu = mu_d_av2[-N[j]:]
        mu_d = []
        for ele in cp_R:
            R_d.append(ele)
            R_d.append(ele)
        R_d_av = np.array(R_d)
        for i in range(1, len(R_d)): R_d_av[i] = (R_d[i] + R_d[i - 1]) / 2
        R_d_av2 = np.array(R_d_av)
        for i in range(1, len(R_d_av)): R_d_av2[i] = (R_d_av[i] + R_d_av[i - 1]) / 2
        cp_R = R_d_av2[-N[j]:]
        R_d = []
    mu = np.exp(cp_mu)
    R = np.exp(cp_R)
    return mu, R


def spo2ida_get_ab_mXXrXXtXX(ac, r, T, pw):
    fb0_mXXrXX = np.matrix([[-0.2226, 0.1401, 0.7604],
                            [-0.0992, -0.0817, -0.1035],
                            [-0.4537, -0.5091, -0.5235],
                            [-0.0398, -0.0236, -0.0287],
                            [0.0829, -0.0364, -0.0174],
                            [0.0193, -0.0126, -0.0118],
                            [-0.1831, -0.2732, -0.5651],
                            [-0.0319, 0.0015, 0.0437],
                            [0.1461, 0.1101, 0.0841],
                            [-0.0227, -0.0045, 0.0159],
                            [-0.0108, 0.0333, 0.0033],
                            [-0.0081, -0.0000, 0.0033],
                            [0.1660, 0.1967, 0.0929],
                            [-0.0124, -0.0304, 0.0130],
                            [0.0273, 0.0396, 0.0580],
                            [-0.0167, -0.0209, -0.0144],
                            [-0.0182, 0.0311, 0.0221],
                            [-0.0097, -0.0047, 0.0007]])
    fb1_mXXrXX = np.matrix([[1.0595, 1.0635, 1.0005],
                            [0.0236, 0.0177, 0.0283],
                            [0.1237, 0.1466, 0.1607],
                            [0.0111, 0.0048, -0.0004],
                            [-0.0023, 0.0102, 0.0021],
                            [0.0008, 0.0019, 0.0035],
                            [-0.0881, -0.1044, -0.1276],
                            [-0.0077, -0.0137, -0.0413],
                            [-0.0239, -0.0090, -0.0085],
                            [0.0025, -0.0014, -0.0198],
                            [0.0082, -0.0003, 0.0037],
                            [0.0007, -0.0013, -0.0043],
                            [0.0317, 0.0038, 0.0673],
                            [0.0006, 0.0065, 0.0074],
                            [-0.0173, -0.0484, -0.0737],
                            [0.0056, 0.0068, 0.0255],
                            [0.0007, -0.0112, -0.0073],
                            [0.0004, 0.0008, 0.0005]])
    X = np.matrix([1, np.log(ac), np.log(r), np.log(ac) * np.log(r), np.power(np.log(r), -1),
                   np.log(ac) * np.power(np.log(r), -1), np.log(T), np.log(ac) * np.log(T), np.log(r) * np.log(T),
                   np.log(ac) * np.log(r) * np.log(T), np.power(np.log(r), -1) * np.log(T),
                   np.log(ac) * np.power(np.log(r), -1) * np.log(T), np.power(np.log(T), 2),
                   np.log(ac) * np.power(np.log(T), 2), np.log(r) * np.power(np.log(T), 2),
                   np.log(ac) * np.log(r) * np.power(np.log(T), 2), np.power(np.log(r), -1) * np.power(np.log(T), 2),
                   np.log(ac) * np.power(np.log(r), -1) * np.power(np.log(T), 2)])
    b0 = X * fb0_mXXrXX
    b1 = X * fb1_mXXrXX
    b0 = b0.tolist()[0]
    b1 = b1.tolist()[0]
    return b0, b1


def spo2ida_get_pinch50_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc):
    Rmc = np.array(Rmc)
    f_mXXtXX = np.matrix([[0.2391, 0.3846, 0.5834],
                          [0.0517, 0.0887, 0.1351],
                          [-1.2399, -1.3531, -1.4585],
                          [-0.0976, -0.1158, -0.1317],
                          [0.0971, 0.1124, 0.1100],
                          [0.0641, 0.0501, 0.0422],
                          [-0.0009, 0.0041, 0.0056],
                          [0.0072, 0.0067, 0.0074]])
    f_p0mXXcXXtXX = np.matrix([[-0.2508, -0.2762, -0.2928],
                               [-0.5517, -0.1992, -0.4394],
                               [0.0941, -0.0031, 0.0683],
                               [0.0059, 0.0101, 0.0131],
                               [0.1681, 0.2451, 0.1850],
                               [0.1357, -0.0199, 0.1783],
                               [-0.0127, 0.0091, -0.0305],
                               [0.0010, -0.0075, -0.0066],
                               [-0.1579, -0.0135, 0.0027],
                               [0.2551, -0.0841, 0.0447],
                               [-0.0602, 0.0222, -0.0151],
                               [0.0087, -0.0003, -0.0025]])
    X = np.matrix(
        [1, np.log(T), np.log(ac), np.log(ac) * np.log(T), np.power(np.log(ac), 2), np.power(np.log(ac), 2) * np.log(T),
         np.power(np.log(ac), 3), np.power(np.log(ac), 3) * np.log(T)])
    Rc_mXX = ac * np.exp(X * f_mXXtXX)
    Rc_mXX = np.array(Rc_mXX.tolist()[0][::-1])
    X = np.matrix([np.log(meq), ac * np.log(meq), np.power(ac, 2) * np.log(meq), np.power(ac, -1) * np.log(meq),
                   np.log(meq) * np.log(T), ac * np.log(meq) * np.log(T), np.power(ac, 2) * np.log(meq) * np.log(T),
                   np.power(ac, -1) * np.log(meq) * np.log(T), np.log(meq) * np.power(np.log(T), 2),
                   ac * np.log(meq) * np.power(np.log(T), 2), np.power(ac, 2) * np.log(meq) * np.power(np.log(T), 2),
                   np.power(ac, -1) * np.log(meq) * np.power(np.log(T), 2)])
    Rfrac_p0mXXcXX = np.exp(X * f_p0mXXcXXtXX)
    Rfrac_p0mXXcXX = Rfrac_p0mXXcXX.tolist()[0][::-1]
    Rfrac_pXXmXXcXX = np.array(Rfrac_p0mXXcXX) + a * (mpeak - np.array(Rfrac_p0mXXcXX))
    Rcap = Rmc + (Rc_mXX - 1) * Rfrac_pXXmXXcXX
    return Rcap, Rc_mXX


def spo2ida_get_mclough_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc):
    Rmc = np.array(Rmc)
    f_mXXtXX = np.matrix([[0.2573, 0.3821, 0.5449],
                          [0.0496, 0.0753, 0.0977],
                          [-1.2305, -1.3289, -1.4270],
                          [-0.0739, -0.0894, -0.1035],
                          [0.0780, 0.0929, 0.1060],
                          [0.0452, 0.0392, 0.0467],
                          [-0.0038, -0.0005, 0.0039],
                          [0.0019, 0.0027, 0.0058]])
    f_p0mXXcXX = np.matrix([[-0.5111, -0.3817, -0.4118],
                            [-0.6194, -0.3599, -0.2610],
                            [0.0928, -0.0019, -0.0070],
                            [0.0163, 0.0186, 0.0158]])
    X = np.matrix(
        [1, np.log(T), np.log(ac), np.log(ac) * np.log(T), np.power(np.log(ac), 2), np.power(np.log(ac), 2) * np.log(T),
         np.power(np.log(ac), 3), np.power(np.log(ac), 3) * np.log(T)])
    Rc_mXX = ac * np.exp(X * f_mXXtXX)
    Rc_mXX = np.array(Rc_mXX.tolist()[0][::-1])
    X = np.matrix([np.log(meq), ac * np.log(meq), np.power(ac, 2) * np.log(meq), np.power(ac, -1) * np.log(meq)])
    Rfrac_p0mXXcXX = np.exp(X * f_p0mXXcXX)
    Rfrac_p0mXXcXX = np.array(Rfrac_p0mXXcXX.tolist()[0][::-1])
    Rfrac_pXXmXXcXX = np.array(Rfrac_p0mXXcXX) + a * (mpeak - np.array(Rfrac_p0mXXcXX))
    Rcap = Rmc + (Rc_mXX - 1) * Rfrac_pXXmXXcXX
    return Rcap, Rc_mXX


def spo2ida_get_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc, pw):
    pRcap, pRc_mXX = spo2ida_get_pinch50_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc)
    mRcap, mRc_mXX = spo2ida_get_mclough_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc)
    Rcap = pw * np.array(pRcap) + (1 - pw) * np.array(mRcap)
    Rc_mXX = pw * np.array(pRc_mXX) + (1 - pw) * np.array(mRc_mXX)
    return Rcap, Rc_mXX


def spo2ida_get_pinch50_ab_pXXtXX(a, T):
    fb0_pXXtXX = np.matrix([[-0.9309, 0.0288, 0.2987],
                            [0.43, -0.1718, 0.0438],
                            [-0.2934, 0.1189, -0.1008],
                            [0.3409, -0.0986, -0.0267],
                            [0.7201, 0.8073, 0.0962],
                            [0.3105, -0.2548, 0.3569],
                            [-0.3343, -0.0561, -0.4138],
                            [0.0778, -0.343, -0.151],
                            [0.1358, -0.85, -0.4004],
                            [-0.7301, 0.4165, -0.3644],
                            [0.6055, -0.0806, 0.4698],
                            [-0.4094, 0.4322, 0.1895]])
    fb1_pXXtXX = np.matrix([[0.1151, -0.4671, -0.5994],
                            [-0.0940, 0.4071, 0.2858],
                            [0.0539, -0.2373, -0.1310],
                            [0.0073, 0.4093, 0.4984],
                            [-0.4092, -1.0761, -0.9235],
                            [0.1534, 0.7899, 0.5074],
                            [0.0216, -0.2518, -0.0910],
                            [0.1651, 0.6914, 0.7160],
                            [0.2733, 1.5106, 1.5379],
                            [-0.0338, -1.1673, -0.7999],
                            [-0.0801, 0.4927, 0.2387],
                            [-0.1586, -1.0815, -1.2277]])
    X = np.matrix([1, np.log(T), np.power(np.log(T), 2), np.power(np.log(T + 1), -1), a, a * np.log(T),
                   a * np.power(np.log(T), 2), a * np.power(np.log(T + 1), -1), np.sqrt(a), np.sqrt(a) * np.log(T),
                   np.sqrt(a) * np.power(np.log(T), 2), np.sqrt(a) * np.power(np.log(T + 1), -1)])
    b0 = np.exp(X * fb0_pXXtXX)
    b1 = np.exp(X * fb1_pXXtXX) - 1
    return b0, b1


def spo2ida_get_mclough_ab_pXXtXX(a, T):
    fb0_pXXtXX = np.matrix([[-0.6157, -0.1842, 0.2155],
                            [0.0260, -0.0027, 0.0760],
                            [0.0140, 0.0215, -0.1458],
                            [0.8605, 0.4986, 0.1693],
                            [0.2264, 0.2026, 0.4712],
                            [-0.3041, -0.3542, -0.7131],
                            [-0.3316, -0.3536, -0.3827],
                            [-0.2613, -0.2011, -0.5240],
                            [0.2689, 0.3055, 0.8093]])
    fb1_pXXtXX = np.matrix([[0.1433, 0.0882, 0.0552],
                            [-0.1074, -0.1635, -0.3562],
                            [0.0538, 0.1062, 0.2745],
                            [-0.1705, -0.1486, -0.1009],
                            [-0.0813, -0.1489, -0.3903],
                            [0.1661, 0.2618, 0.5810],
                            [0.0313, 0.0587, 0.0363],
                            [0.1957, 0.3185, 0.7552],
                            [-0.2086, -0.3550, -0.8369]])
    X = np.matrix([1, np.log(T), np.power(np.log(T), 2), a, a * np.log(T), a * np.power(np.log(T), 2), np.sqrt(a),
                   np.sqrt(a) * np.log(T), np.sqrt(a) * np.power(np.log(T), 2)])
    b0 = np.exp(X * fb0_pXXtXX)
    b1 = np.exp(X * fb1_pXXtXX) - 1
    return b0, b1


def spo2ida_get_ab_pXXtXX(a, T, pw):
    pb0, pb1 = spo2ida_get_pinch50_ab_pXXtXX(a, T)
    mb0, mb1 = spo2ida_get_mclough_ab_pXXtXX(a, T)
    b0 = pw * pb0 + (1 - pw) * mb0
    b1 = pw * pb1 + (1 - pw) * mb1
    b0 = b0.tolist()[0]
    b1 = b1.tolist()[0]
    return b0, b1


def spo2ida_get_Rmc(mc, b0, b1):
    b0 = np.array(b0)
    b1 = np.array(b1)
    Rmc = []
    slmc = []
    for i in range(3):
        if b1[i] != 0:
            Delta = np.power(b0[i], 2) + 4 * b1[i] * np.log(mc)
            lRmc1 = (np.divide(-b0[i] + np.sqrt(Delta), 2 * b1[i]))
        else:
            lRmc1 = (np.divide(np.log(mc), b0[i]))
        Rmc.append(np.exp(lRmc1))
        slmc.append(b0[i] + b1[i] * 2 * lRmc1)
    return Rmc, slmc


def model_pXX(idacm, idacr, a, mc, T, pw, N):
    b0, b1 = spo2ida_get_ab_pXXtXX(a, T, pw)
    Rmc, slmc = spo2ida_get_Rmc(mc, b0, b1)
    for i in range(3):
        RpXX = np.linspace(1, Rmc[i], N + 1)
        RpXX = RpXX[1:]
        idacr[i] = idacm[i] + RpXX.tolist()
        newMu = np.exp(b0[i] * np.log(RpXX) + b1[i] * np.power(np.log(RpXX), 2))
        idacm[i] = idacm[i] + newMu.tolist()
    return idacm, idacr, Rmc, slmc


def model_mXX(idacm, idacr, a, ac, mc, T, pw, mr, meq, mend, mpeak, mf, Rmc, slmc, filletstyle, N):
    Rcap, Rcap_mXX = spo2ida_get_Rcap_pXXmXXcXXtXX(a, ac, T, meq, mpeak, Rmc, pw)
    b0, b1 = spo2ida_get_ab_pXXtXX(0, T, pw)
    for i in range(3):
        if filletstyle == 0 or filletstyle == 1 or filletstyle == 2:
            RmXX = np.linspace(Rmc[i], Rcap[i], N + 1)
            RmXX = RmXX[1:]
            newMu = mc + (RmXX - Rmc[i]) * slmc[i]
            if newMu[-1] > mf:
                f = np.nonzero(newMu <= mf)
                RmXX = RmXX[f]
                newMu = mc + (RmXX - Rmc[i]) * slmc[i]
            idacr[i] = idacr[i] + RmXX.tolist()
            idacm[i] = idacm[i] + newMu.tolist()
        else:
            xi = (np.log(Rcap[i]) - np.log(Rmc[i])) * slmc[i] + np.log(mc)
            cp_mu = [2 * np.log(mc) - xi, xi, 2 * np.log(mend) - xi]
            cp_R = [2 * np.log(Rmc[i]) - np.log(Rcap[i]), np.log(Rcap[i]), np.log(Rcap[i])]
            [newcx, newcy] = spline(cp_mu, cp_R)
            x_mc = np.nonzero(newcx > mc)[0]
            indy = [ele for ele in x_mc if newcx[ele] <= mr]
            m_rXX = mr
            if len(indy) == 0:
                i1 = np.max(np.nonzero(newcx <= mc))
                i2 = i1 + 1
            else:
                if indy[-1] == len(newcx) - 1:
                    i1 = indy[-2]
                    i2 = indy[-1]
                else:
                    i1 = indy[-1]
                    i2 = indy[-1] + 1
            R_rXX = newcy[i2] + (newcy[i2] - newcy[i1]) / (newcx[i2] - newcx[i1]) * (mr - newcx[i2])
            idacm[i] = idacm[i] + newcx[indy].tolist() + [m_rXX]
            idacr[i] = idacr[i] + newcy[indy].tolist() + [R_rXX]
    return idacm, idacr


def model_rXX(idacm, idacr, a, ac, mc, mr, mf, r, req, T, pw, filletstyle, N):
    b0, b1 = spo2ida_get_ab_mXXrXXtXX(ac, req, T, pw)
    for i in range(3):
        Rmr = idacr[i][-1]
        real_mi = max(np.exp(b0[i] + np.log(Rmr) * b1[i]), mr)
        mi = min(mf, real_mi)
        if filletstyle == 0:
            m_rXX = np.linspace(mi, mf, N + 1)
            if mi < mf:
                R_rXX = np.exp((np.log(m_rXX) - b0[i]) / b1[i])
            else:
                R_rXX = np.repeat(Rmr, len(m_rXX))
            idacr[i] = idacr[i] + R_rXX.tolist()
            idacm[i] = idacm[i] + m_rXX.tolist()
        else:
            if round(np.log(idacr[i][-1]) - np.log(idacr[i][-2]), 3) != 0:
                slope_mXXend = (np.log(idacm[i][-1]) - np.log(idacm[i][-2])) / (
                            np.log(idacr[i][-1]) - np.log(idacr[i][-2]))
                if slope_mXXend == b1[i]: slope_mXXend = b1[i] + 0.05
                int_mXXend = np.log(mr) - slope_mXXend * np.log(Rmr)
                lRmi = (int_mXXend - b0[i]) / (b1[i] - slope_mXXend)
            else:
                lRmi = np.log(Rmr)
            if lRmi < np.log(Rmr):
                if b1[i] >= slope_mXXend:
                    slope_mXXend = b1[i] + 0.05
                else:
                    slope_mXXend = b1[i] - 0.05
                int_mXXend = np.log(mr) - slope_mXXend * np.log(Rmr)
                lRmi = (int_mXXend - b0[i]) / (b1[i] - slope_mXXend)
            lmmi = b0[i] + b1[i] * lRmi
            new_mmi = np.exp(lmmi)
            cp_mu = [2 * np.log(mr) - lmmi, lmmi, 3 * lmmi]
            cp_R = [2 * np.log(Rmr) - lRmi, lRmi, (3 * lmmi - b0[i]) / b1[i]]
            newcx, newcy = spline(cp_mu, cp_R)
            x_mc = np.nonzero(newcx > mr)[0]
            indy = [ele for ele in x_mc if newcx[ele] <= mf]
            if mf > np.power(new_mmi, 2):
                m_rXX = np.linspace(new_mmi ** 2, mf, N + 1)
                m_rXX = m_rXX
                R_rXX = np.exp((np.log(m_rXX) - b0[i]) / b1[i])
            else:
                m_rXX = mf
                if len(indy) == 0:
                    i1 = np.max(np.nonzero(newcx <= mr))
                    i2 = i1 + 1
                else:
                    if indy[-1] == len(newcx) - 1:
                        i1 = indy[-2]
                        i2 = indy[-1]
                    else:
                        i1 = indy[-1]
                        i2 = indy[-1] + 1
                R_rXX = newcy[i2] + (newcy[i2] - newcy[i1]) / (newcx[i2] - newcx[i1]) * (mf - newcx[i2])
            idacm[i] = idacm[i] + newcx[indy].tolist() + [m_rXX]
            idacr[i] = idacr[i] + newcy[indy].tolist() + [R_rXX]
    return idacm, idacr


def regions2model(a, mc, ac, r, mf, mr):
    if (mc > 1) and (mf > 1):
        pxx = 1
    else:
        pxx = 0
    if (ac != 0) and (mf > mc) and (mr > mc):
        mxx = 1
    else:
        mxx = 0
    if (r != 0) and (mf > mr):
        rxx = 1
    else:
        rxx = 0
    return pxx, mxx, rxx


def spo2ida_allT(mc, a, ac, r, mf, T, pw, filletstyle=3, N=10):
    # mc      : mu of end of hardening slope
    # a       : hardening slope in [0,1]
    # ac      : negative (capping) slope in [0.02,4]
    # r       : Residual plateau height, a fraction of Fy
    # mf      : fracturing mu (end of SPO)
    # T       : period (sec)
    # pw      : pinching model weight.
    mc = float(mc)
    a = float(a)
    ac = abs(float(ac))
    r = float(r)
    mf = float(mf)
    T = float(T)
    pw = float(pw)
    error_flag = 0
    if mc < 1 or mc > 9:
        print('ERROR: We must have "mc" within [1,9]')
        error_flag = 1
    if a < 0 or a > 0.90:
        print('ERROR: We must have "a" within [0,0.90]')
        error_flag = 1
    if ac < 0.02 or ac > 4:
        print('ERROR: We must have "ac" within [0.02,4]')
        error_flag = 1
    if r < 0 or r > 0.95:
        print('ERROR: We must have "r" within [0,0.95]')
        error_flag = 1
    if mf < 1:
        print('ERROR: We must have "mf" > 1')
        error_flag = 1
    if T < 0.1 or T > 4:
        print('ERROR: We must have "T" within [0.1,4]')
        error_flag = 1
    if error_flag == 1: return
    mr = mc + (1 + (mc - 1) * a - r) / ac
    rpeak = 1 + a * (mc - 1)
    mend = mc + rpeak / ac
    meq = mend - 1 / ac
    Rpeak = 1 + min(a, 0.05) * (mc - 1)
    req = r / Rpeak
    mpeak = mend * ac / (1 + ac)
    pxx, mxx, rxx = regions2model(a, mc, ac, r, mf, mr)
    mc = min(mc, mf)
    mr = min(mr, mf)
    idacm = [np.linspace(0, 1, N + 1).tolist()] * 3
    idacr = [np.linspace(0, 1, N + 1).tolist()] * 3
    if pxx:
        [idacm, idacr, Rmc, slmc] = model_pXX(idacm, idacr, a, mc, T, pw, N)
    else:
        Rmc, slmc = [1, 1, 1], [1, 1, 1]
    if mxx: [idacm, idacr] = model_mXX(idacm, idacr, a, ac, mc, T, pw, mr, meq, mend, mpeak, mf, Rmc, slmc, filletstyle,
                                       N)
    if rxx: [idacm, idacr] = model_rXX(idacm, idacr, a, ac, mc, mr, mf, r, req, T, pw, filletstyle, N)
    for i in range(3):
        idacr[i] = idacr[i] + [idacr[i][-1]] + [idacr[i][-1]]
        idacm[i] = idacm[i] + [mf, mf + 2]
    R84, R50, R16 = idacr[0][-1], idacr[1][-1], idacr[2][-1]
    idas = idacm + idacr
    spom, spor = spo2ida_spo(mc, a, ac, r, mf)
    return R16, R50, R84, idacm, idacr, spom, spor


def spo2ida_spo(mc, a, ac, r, mf):
    ac = abs(ac)
    spocm = np.zeros(5)
    spocr = np.zeros(5)
    spocm[1] = 1
    spocr[1] = 1
    spocm[2] = min(mc, mf)
    spocr[2] = 1 + (spocm[2] - 1) * a
    spocm[3] = min(mf, spocm[2] - (r - spocr[2]) / ac)
    if r == 0 and mf > spocm[3]:
        spocr[3] = 0
    else:
        spocr[3] = spocr[2] - (spocm[3] - spocm[2]) * ac
    spocr[4] = r
    spocm[4] = mf
    return spocm, spocr


def plot_spo2ida(mc, a, ac, r, mf, T, pw):
    spom, spor = spo2ida_spo(mc, a, ac, r, mf)
    R16, R50, R84, idacm, idacr, spom, spor = spo2ida_allT(mc, a, ac, r, mf, T, pw)
    colors = ['r', 'r', 'r']
    lss = [':', '-', ':']
    tick_fontsize, axis_label_fontsize, legend_fontsize = 16, 18, 16
    f, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(spom, spor, color='b', zorder=10)
    ax.fill_between(spom, 0, spor, color='lightcyan')
    for idx, i in enumerate(idacm): ax.plot(idacm[idx], idacr[idx], color=colors[idx], ls=lss[idx])
    splineA = interp1d(idacm[0], idacr[0])
    splineB = interp1d(idacm[-1], idacr[-1])
    xnew = np.linspace(idacm[0][0], idacm[0][-1], 1000)
    ax.fill_between(xnew, splineA(xnew), splineB(xnew), color='mistyrose')
    xticks = np.arange(0, 7 + 1, 1)
    yticks = xticks
    x_max = xticks[-1]
    y_max = yticks[-1]
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.0f' % (i) for i in xticks], fontsize=tick_fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['%.0f' % (i) for i in yticks], fontsize=tick_fontsize)
    ax.grid(color='0.75')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(length=5)
    ax.yaxis.set_tick_params(length=5)
    ax.set_xlabel(r'$\mathrm{\mu}$', fontsize=axis_label_fontsize)
    ax.set_ylabel(r'$\mathrm{R}$', fontsize=axis_label_fontsize)
    f.tight_layout()


if __name__ == "__main__":
    # Input
    mc = 3
    a = 0.01
    ac = -1.0
    r = 0.1
    mf = 6
    pw = 1
    T = 0.8
    # Run

    R16, R50, R84, idacm, idacr, spom, spor = spo2ida_allT(mc, a, ac, r, mf, T, pw)
