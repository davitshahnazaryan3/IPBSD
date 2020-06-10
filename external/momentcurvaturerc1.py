"""
                                    CUMBIARECT

               SECTION AND MEMBER RESPONSE OF RC MEMBERS OF RECTANGULAR SECTION

                       LUIS A. MONTEJO (lumontv@yahoo.com.ar)

                 uptades available at www.geocities.com/lumontv/eng

            DEPARTMENT OF CIVIL, CONSTRUCTION AND ENVIROMENTAL ENGINEERING

                         NORTH CAROLINA STATE UNIVERSITY

Adopted from the matlab script by D.S.
"""
import timeit
import numpy as np
from scipy.interpolate import interp1d


class MomentCurvatureRC1:

    def __init__(self, b, h, ncx, ncy, cover, Mtarget, L, P=0., nlayers=0, bending='single', Dh=8, s=100, fc_prime=25,
                 fy=415, Ec=0, Es=200000., k_hard=1.35, confined='mc', unconfined='mu', rebar='ks',
                 ductility_mode='uniaxial', pflag=False, dflag=False, rflag=False):
        """
        Initialization
        :param b: float                     Width of section in mm
        :param h: float                     Height of section in mm
        :param ncx: int                     Transverse steel legs in x direction (confinement)
        :param ncy: int                     Transverse steel legs in y direction (shear)
        :param cover: float                 Clear cover to transverse bars in mm
        :param Mtarget: float               Target bending moment in kNm
        :param L: float                     Length of element in mm
        :param P: float                     Axial load in kN (-) tension (+) compression
        :param nlayers: int                 Number of reinforcement layers
        :param bending: str                 'single' or 'double' bending
        :param Dh: int                      Diameter of transverse reinf. in mm
        :param s: float                     Spacing of transverse reinf. in mm
        :param fc_prime: float              Concrete compressive strength in MPa
        :param fy: float                    Steel yielding stress in MPa
        :param Ec: float                    Concrete modulus of elasticity (0 for automatic calculation)
        :param Es: float                    Steel modulus of elasticity
        :param k_hard: float                Steel peak to yield ratio
        :param confined: str                Confined concrete material model ('mc' or 'mu'), for lightweight 'mclw'
        :param unconfined: str              Unconfined concrete material model ('mc' or 'mu'), for lightweight 'mclw'
        :param rebar: str                   Rebar material model ('ks' for King, and 'ra' for Raynor)
        :param ductility_mode: str          Ductility mode, 'biaxial' or 'uniaxial'
        :param pflag: bool                  Flag for plotting
        :param dflag: bool                  Flag for displaying information
        :param rflag: bool                  Flag for recording and storing data
        """
        self.b = b
        self.h = h
        self.ncx = ncx
        self.ncy = ncy
        self.cover = cover
        self.Mtarget = Mtarget*10**6        # into Nmm
        self.L = L
        self.P = P*1000                     # into Newtons
        self.nlayers = nlayers
        self.bending = bending
        self.Dh = Dh
        self.s = s
        self.fc_prime = fc_prime
        self.fy = fy
        self.Ec = Ec
        self.Es = Es
        self.k_hard = k_hard
        self.confined = confined
        self.unconfined = unconfined
        self.rebar = rebar
        self.clb = self.cover + self.Dh
        self.ductility_mode = ductility_mode
        self.pflag = pflag
        self.dflag = dflag
        self.rflag = rflag

        # Material properties, Constants
        # (todo, make them as input parameters depending on class of concrete or reinforcement)
        # Concrete
        self.eco = 0.002
        self.esm = 0.08
        self.espall = 0.0064

        # Steel
        self.fsu = self.fy*self.k_hard
        self.esh = 0.00276
        self.esu = 0.08

        # Temperature information
        self.TEMP = 30
        self.kLsp = 0.022

        # Control parameters
        self.ITERMAX = 1000
        self.NCL = 40
        self.TOL = 0.001
        self.DELS = 0.0001

        # Deformation limit states
        self.ECSER = 0.004
        self.ESSER = 0.015
        self.ECDAM = "twth"
        self.ESDAM = 0.06

        # Strain limits for interaction diagram
        self.CSID = 0.004
        self.SSID = 0.015

        # Slope of the yield plateau (MPa)
        self.Ey = 700.
        # defines strain hardening curve in the Raynor model [2-6]
        self.C1 = 3.3

    @staticmethod
    def get_init_time():
        """
        Records initial time
        :return: float                      Initial time
        """
        start_time = timeit.default_timer()
        return start_time

    @staticmethod
    def truncate(n, decimals=0):
        """
        Truncates time with given decimal points
        :param n: float                     Time
        :param decimals: int                Decimal points
        :return: float                      Truncated time
        """
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def get_time(self, start_time):
        """
        Prints running time in seconds and minutes
        :param start_time: float            Initial time
        :return: None
        """
        elapsed = timeit.default_timer() - start_time
        print('Running time: ', self.truncate(elapsed, 1), ' seconds')
        print('Running time: ', self.truncate(elapsed / float(60), 2), ' minutes')

    def get_MLR(self, area):
        """
        Gets longitudinal reinforcement details (assumes one diameter only)
        Assuming symmetric sections for columns
        :param area: float                      Total reinforcement area
        :return: ndarray                        Longitudinal reinforcement details, e.g. [cover to center, N rebars,
                                                Diameter]
        """
        def get_idx_high(target, data):
            if np.where(data >= target)[0].size == 0:
                return np.nan
            else:
                return np.where(data >= target)[0][0]

        diameter = np.array([8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32])
        D = [40]
        if self.nlayers == 0:
            area = np.array([area / 2, area / 2])
            n = 4
            while D[0] not in diameter:
                n = np.array([n/2, n/2])
                D = diameter[get_idx_high(np.sqrt(4*area[0]/np.pi/n[0]), diameter)]
                n += 2
            z = np.array([self.h - self.clb - D/2, self.clb + D/2])
            n = np.array([n/2, n/2], dtype='int')
            D = np.array([D]*len(n))
        elif self.nlayers == 1:
            area = np.array([area * 3 / 8, area * 2 / 8, area * 3 / 8])
            n = np.array([3, 2, 3], dtype='int')
            D = diameter[get_idx_high(np.sqrt(4*area[0]/np.pi/n[0]), diameter)]
            if D not in diameter:
                print(f"[WARNING] Diameter of {D} is larger than 32mm...")
            z = np.array([self.h - self.clb - D/2, self.h / 2, self.clb + D/2])
            D = np.array([D]*len(n))
        elif self.nlayers == 2:
            area = np.array([area * 4 / 12, area * 2 / 12, area * 2 / 12, area * 4 / 12])
            n = np.array([4, 2, 2, 4], dtype='int')
            D = diameter[get_idx_high(np.sqrt(4*area[0]/np.pi/n[0]), diameter)]
            if D not in diameter:
                print(f"[WARNING] Diameter of {D} is larger than 32mm...")
            z = np.array([round(self.h - self.clb - D/2, 0),
                          round(((self.h - 2 * (self.clb + D/2)) / (1 + self.nlayers) + self.clb + D/2) * 2 -
                                (self.clb + D/2), 0),
                          round(((self.h - 2 * (self.clb + D/2)) / (1 + self.nlayers) + (self.clb + D/2)), 0),
                          round((self.clb + D/2), 0)])
            D = np.array([D]*len(n))
        else:
            raise ValueError(f"[EXCEPTION] wrong input {self.nlayers} of number of reinforcement layers")
        MLR = (np.vstack((z, n, D))).T
        return MLR[MLR[:, 0].argsort()]

    def manderun(self, Ec, fpc, eco, espall, dels):
        """
        Unconfined concrete, mander model
        :param Ec: float                        Elastic modulus of concrete
        :param fpc: float
        :param eco:
        :param espall:
        :param dels:
        :return:
        """
        ec = np.arange(0, espall + dels, dels)
        Esecu = fpc / eco
        ru = Ec / (Ec - Esecu)
        xu = ec / eco
        fcu = np.zeros(len(ec))

        for i in range(len(ec)):
            if ec[i] < 2 * eco:
                fcu[i] = fpc * xu[i] * ru / (ru - 1 + xu[i] ** ru)
            elif 2 * eco <= ec[i] <= espall:
                fcu[i] = fpc * (2 * ru / (ru - 1 + 2 ** ru)) * (1 - (ec[i] - 2 * eco) / (espall - 2 * eco))
            elif ec[i] > espall:
                fcu[i] = 0
        return ec, fcu

    def manderconf(self, Ec, Ast, Dh, clb, s, fpc, fy, eco, esm, section, d, b, ncx, ncy, wi, dels):
        # confined concrete
        sp = s - Dh
        Ash = 0.25 * np.pi * Dh ** 2

        if section == "rectangular":
            bc = b - 2 * clb + Dh
            dc = d - 2 * clb + Dh
            Asx = ncx * Ash
            Asy = ncy * Ash
            Ac = bc * dc
            rocc = Ast / Ac
            rox = Asx / (s * dc)
            roy = Asy / (s * bc)
            ros = rox + roy
            ke = ((1 - sum(wi ** 2) / (6 * bc * dc)) * (1 - sp / (2 * bc)) * (1 - sp / (2 * dc))) / (1 - rocc)
            ro = 0.5 * ros
            fpl = ke * ro * fy

        else:
            raise ValueError("[EXCEPTION] Section not available")

        fpcc = (-1.254 + 2.254 * np.sqrt(1 + 7.94 * fpl / fpc) - 2 * fpl / fpc) * fpc
        ecc = eco * (1 + 5 * (fpcc / fpc - 1))
        Esec = fpcc / ecc
        r = Ec / (Ec - Esec)
        ecu = round(1.4 * (0.004 + 1.4 * ros * fy * esm / fpcc), 4)
        ec = np.arange(0, ecu + dels, dels)
        x = 1 / ecc * ec
        fc = fpcc * x * r / (r - 1 + x ** r)
        return ec, fc

    def manderconflw(self, Ec, Ast, Dh, clb, s, fpc, fy, eco, esm, espall, section, d, b, ncx, ncy, wi, dels):
        # confined lightweight concrete
        sp = s - Dh
        Ash = 0.25 * np.pi * Dh ** 2

        if section == "rectangular":
            bc = b - 2 * clb + Dh
            dc = d - 2 * clb + Dh
            Asx = ncx * Ash
            Asy = ncy * Ash
            Ac = bc * dc
            rocc = Ast / Ac
            rox = Asx / (s * dc)
            roy = Asy / (s * bc)
            ros = rox + roy
            ke = ((1 - sum(wi ** 2) / (6 * bc * dc)) * (1 - sp / (2 * bc)) * (1 - sp / (2 * dc))) / (1 - rocc)
            ro = 0.5 * ros
            fpl = ke * ro * fy

        else:
            raise ValueError("[EXCEPTION] Section not available")

        fpcc = (1 + fpc / (2 * fpc)) * fpc
        ecc = eco * (1 + 5 * (fpcc / fpc - 1))
        Esec = fpcc / ecc
        r = Ec / (Ec - Esec)
        ecu = round(1.5 * (0.004 + 1.4 * ros * fy * esm / fpcc), 4)

        ec = np.arange(0, ecu + dels, dels)
        x = 1 / ecc * ec
        fc = fpcc * x * r / (r - 1 + x ** r)
        return ec, fc

    def steelking(self, Es, fy, fsu, esh, esu, dels):
        r = esu - esh
        m = ((fsu / fy) * ((30 * r + 1) ** 2) - 60 * r - 1) / (15 * r ** 2)
        es = np.arange(0, esu + dels, dels)
        ey = fy / Es
        fs = np.zeros(len(es))

        for i in range(len(es)):
            if es[i] < ey:
                fs[i] = Es * es[i]
            elif ey <= es[i] <= esh:
                fs[i] = fy
            elif es[i] > esh:
                fs[i] = ((m * (es[i] - esh) + 2) / (60 * (es[i] - esh) + 2) +
                         (es[i] - esh) * (60 - m) / (2 * ((30 * r + 1) ** 2))) * fy
        return es, fs

    def raynor(self, Es, fy, fsu, esh, esu, dels, C1, Ey):
        es = np.arange(0, esu + dels, dels)
        ey = fy / Es
        fsh = fy + (esh - ey) * Ey
        fs = np.zeros(len(es))

        for i in range(len(es)):
            if es[i] < ey:
                fs[i] = Es * es[i]
            elif ey <= es[i] <= esh:
                fs[i] = fy + (es[i] - ey) * Ey
            elif es[i] > esh:
                fs[i] = fsu - (fsu - fsh) * (((esu - es[i]) / (esu - esh)) ** C1)
        return es, fs

    def get_cracking(self):
        """
        Calculates properties of concrete
        :return: float                          Concrete strain for cracking
        """
        if self.Ec == 0:
            self.Ec = 3320 * np.sqrt(self.fc_prime) + 6900

        # Tensile strength
        if self.TEMP < 0:
            Ct = (1 - 0.0105*self.TEMP)*0.56*self.fc_prime**0.5
        else:
            Ct = 0.56*self.fc_prime**0.5
        eccr = Ct/self.Ec
        return eccr

    def get_wi(self, MLR):
        """
        Gets a vector with clear distances between periferical longitudinal bars properly restrained
        :param MLR: ndarray                     Longitudinal reinforcement details
        :return: ndarray                        The vector
        """
        if len(MLR) > 1:
            # number of periferical bars
            Pbars = MLR[0][1] + MLR[-1][1] + 2 * (len(MLR) - 2)
            a1 = ((self.b - 2 * self.clb - MLR[0][1] * MLR[0][2]) / (MLR[0][1] - 1)) * np.ones(int(MLR[0][1] - 1))
            a2 = ((self.b - 2 * self.clb - MLR[-1][1] * MLR[-1][2]) / (MLR[-1][1] - 1)) * np.ones(int(MLR[-1][1] - 1))
            a3 = (np.diff(MLR[:, 0]) - np.mean(MLR[:, 2]))
            a4 = (np.diff(MLR[:, 0]) - np.mean(MLR[:, 2]))
            wi = np.concatenate((a1, a2, a3, a4))
            return wi
        elif len(MLR) == 1:
            a1 = (self.h - 2 * self.clb - 2 * MLR[0][2]) * np.ones(2)
            a2 = (self.b - 2 * self.clb - 2 * MLR[0][2]) * np.ones(2)
            wi = np.concatenate((a1, a2))
            return wi

    def get_disp(self, ecu, ecun, fcun, ec, fc, es, fs, yl, conclay):
        """
        Gets displacements with deformations in the top concrete
        :param ecu: float                       Maximum strain of confined concrete
        :param ecun: ndarray                    Strains of unconfined concrete
        :param fcun: ndarray                    Stresses of unconfined concrete
        :param ec: ndarray                      Strains of confined concrete
        :param fc: ndarray                      Stresses of confined concrete
        :param es: ndarray                      Strains of the steel
        :param fs: ndarray                      Stresses of the steel
        :param yl: ndarray                      Layers of unconfined concrete
        :param conclay: conclay                 [center layer|A uncon|A conf|d top layer]
        :return: int, ndarray                   Length of disp array, disp array = deformations in the top concrete
        """
        if ecu <= 0.0018:
            disp = np.arange(0.0001, 20 * ecu + 0.0001, 0.0001)
        elif 0.0018 < ecu <= 0.0025:
            disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 20 * ecu + 0.0002, 0.0002)))
        elif 0.0025 < ecu <= 0.006:
            disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                              np.arange(0.0025, 20 * ecu + 0.0005, 0.0005)))
        elif 0.006 < ecu <= 0.012:
            disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                              np.arange(0.0025, 0.0055, 0.0005), np.arange(0.006, 20 * ecu + 0.001, 0.001)))
        elif ecu > 0.012:
            disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                              np.arange(0.0025, 0.0055, 0.0005), np.arange(0.006, 0.011, 0.001),
                              np.arange(0.012, 20 * ecu + 0.002, 0.002)))

        ndisp = len(disp)

        if self.P > 0:
            for k in range(ndisp):
                interpolator1 = interp1d(ecun, fcun)
                interpolator2 = interp1d(ec, fc)
                interpolator3 = interp1d(es, fs)

                compch = sum(interpolator1(disp[0] * np.ones(len(yl))) * conclay[1, :]) + \
                         sum(interpolator2(disp[0] * np.ones(len(yl))) * conclay[2, :]) + \
                         interpolator3(disp[0]) * Ast

                if compch < self.P:
                    disp = disp[1:len(disp) + 1]

        ndisp = len(disp)
        return ndisp, disp

    def iterate_for_m_phi(self, disp, conclay, distld, Asbs, ecun, fcun, ec, fc, es, fs, dcore, ecu, esu):
        """
        Iteration for constructing the moment-curvature relationship
        :param disp: ndarray                    Deformations in the top concrete
        :param conclay: ndarray                 [center layer|A uncon|A conf|d top layer]
        :param distld: ndarray                  y coordinate of each bar
        :param Asbs: ndarray                    Area of each bar
        :param ecun: ndarray                    Strains of unconfined concrete
        :param fcun: ndarray                    Stresses of unconfined concrete
        :param ec: ndarray                      Strains of confined concrete
        :param fc: ndarray                      Stresses of confined concrete
        :param es: ndarray                      Strains of the steel
        :param fs: ndarray                      Stresses of the steel
        :param dcore: float                     Distance to the core
        :param ecu: float                       Maximum strain of confined concrete
        :param esu: float                       Maximum strain of steel
        :return: ndarrays                       Descriptions within the body
        """
        # stop conditions
        message = 0

        # Initialization of the main parameters
        curv = np.array([0])                    # curvature
        mom = np.array([0])                     # moments
        ejen = np.array([0])                    # neutral axis
        DF = np.array([0])                      # force equilibrium
        vniter = np.array([0])                  # iterations

        coverstrain = np.array([0])
        corestrain = np.array([0])
        steelstrain = np.array([0])

        # tolerance allowed
        tol = self.TOL * self.h * self.b * self.fc_prime
        x = np.array([self.h / 2])
        ndisp = len(disp)

        for k in range(ndisp):
            lostmomcontrol = max(mom)
            if mom[k] < 0.8 * lostmomcontrol:
                message = 4
                break

            F = 10 * tol
            niter = -1
            while abs(F) > tol:
                niter += 1
                if x[niter] <= self.h:
                    # strains in the concrete
                    eec = (disp[k] / x[niter]) * (conclay[0, :] - (self.h - x[niter]))
                    # strains in the steel
                    ees = (disp[k] / x[niter]) * (distld - (self.h - x[niter]))
                else:
                    eec = (disp[k] / x[niter]) * (x[niter] - self.h + conclay[0, :])
                    ees = (disp[k] / x[niter]) * (x[niter] - self.h + distld)

                # stresses in the unconfined cocncrete
                fcunconf = interp1d(ecun, fcun)(eec)
                # stresses in the confined concrete
                fcconf = interp1d(ec, fc)(eec)
                # stresses in the steel
                fsteel = interp1d(es, fs)(ees)
                FUNCON = fcunconf * conclay[1, :]
                FCONF = fcconf * conclay[2, :]
                FST = Asbs * fsteel
                F = sum(FUNCON) + sum(FCONF) + sum(FST) - self.P
                if F > 0:
                    x = np.append(x, x[niter] - 0.05 * x[niter])
                elif F < 0:
                    x = np.append(x, x[niter] + 0.05 * x[niter])

                # force stop
                if niter > self.ITERMAX:
                    message = 3
                    break
                # end of while loop
            cores = (disp[k] / x[niter]) * abs(x[niter] - dcore)
            TF = (self.confined == self.unconfined)
            if not TF:
                if cores >= ecu:
                    message = 1
                    break
            elif TF:
                if disp[k] >= ecu:
                    message = 1
                    break
            if abs(ees[0]) > esu:
                message = 2
                break

            ejen = np.append(ejen, x[niter])
            DF = np.append(DF, F)
            vniter = np.append(vniter, niter)
            mom = np.append(mom, (
                        sum(FUNCON * conclay[0, :]) + sum(FCONF * conclay[0, :]) + sum(FST * distld) + self.P *
                        self.h / 2 / 10 ** 6))
            if mom[k + 1] < 0:
                mom[k + 1] = -0.01 * mom[k + 1]
            curv = np.append(curv, 1000 * disp[k] / x[niter])
            coverstrain = np.append(coverstrain, disp[k])
            corestrain = np.append(corestrain, cores)
            steelstrain = np.append(steelstrain, ees[0])
            x = np.array([x[niter]])
            if message != 0:
                break
        return curv, mom, ejen, DF, vniter, coverstrain, corestrain, steelstrain

    def get_buckling(self, curv, fycurv, eqcurv, curvature_ductility, steelstrain, diam, TransvSteelRatioAverage, Lp,
                     LBE):
        """
        Gets buckling models
        :param curv: ndarray                                Curvature
        :param fycurv: float                                Curvature at first yield
        :param eqcurv: float                                Curvature at nominal yield
        :param curvature_ductility: float                   Curvature ductility
        :param steelstrain: ndarray                         Strains for steel
        :param diam: ndarray                                Diameters of steel bars
        :param TransvSteelRatioAverage: float               Average transverse steel ratio
        :param Lp: float                                    Plastic hinge length
        :param LBE: float                                   Buckling length
        :return: ndarray                                    Curvature ductilities
        """
        # Moyer - Kowalsky Buckling model
        bucritMK = 0
        CuDu = curv / eqcurv

        if curvature_ductility > 4:
            # residual growth strain at ductility 4
            esgr4 = -0.5 * interp1d(CuDu, steelstrain)(4)
            # allowable steel compression strain
            escc = 3 * ((self.s / diam[0]) ** (-2.5))

            esgr = np.zeros(len(steelstrain))
            for i in range(len(steelstrain)):
                if CuDu[i] < 1:
                    esgr[0] = 0
                elif 1 < CuDu[i] < 4:
                    esgr[i] = esgr4 / 4 * CuDu[i]
                elif CuDu[i] > 4:
                    esgr[i] = -0.5 * steelstrain[i]
            esfl = escc - esgr

            if -steelstrain[-1] >= esfl[-1]:
                bucritMK = 1
                fail = esfl - (-steelstrain)
                failCuDuMK = interp1d(fail, CuDu)(0)
                failesfl = interp1d(fail, esfl)(0)
                failss = -interp1d(fail, steelstrain)(0)
            else:
                failCuDuMK = None
        else:
            failCuDuMK = None

        # Berry - Eberhard buckling model
        bucritBE = 0
        # model constants
        C0 = 0.019
        C1 = 1.65
        C2 = 1.797
        C3 = 0.012
        C4 = 0.072

        # effective confinement ratio
        roeff = (2 * TransvSteelRatioAverage) * self.fy / self.fc_prime

        # plastic rotation at the onset of bar buckling
        rotb = C0 * (1 + C1 * roeff) * ((1 + C2 * self.P / (self.b * self.h * self.fc_prime)) ** (-1)) * (
                    1 + C3 * LBE / h + C4 * diam[0] * self.fy / self.h)

        plrot = (curv - fycurv) * Lp / 1000

        if max(plrot) > rotb:
            bucritBE = 1
            failBE = plrot - rotb
            failplrot = interp1d(failBE, plrot)(0)
            failCuDuBE = interp1d(failBE, CuDu)(0)
        else:
            failCuDuBE = None

        return CuDu, bucritMK, bucritBE, failCuDuMK, failCuDuBE

    def get_shear_strength(self, LongSteelRatio, cMn, displ, dyf, alpha):
        """
        Calculates shear strength of element
        :param LongSteelRatio: float                    Longitudinal steel ratio
        :param cMn: float                               Nominal yield strength
        :param displ: ndarray                           Total displacement from shear and flexure contributions
        :param dyf: float                               Nominal yield displacement
        :param alpha: float                             Factor for shear strength calculation
        :return: floats                                 Unfactored and factored shear strengths
        """
        # Shear strength
        Vs = (ncy * 0.25 * np.pi * self.Dh ** 2 * self.fy * (self.h - self.clb + 0.5 * self.Dh - cMn) /
              np.tan(np.pi / 6) / self.s) / 1000
        Vsd = (ncy * 0.25 * np.pi * self.Dh ** 2 * self.fy * (h - self.clb + 0.5 * self.Dh - cMn) /
               np.tan(35 / 180 * np.pi) / self.s) / 1000
        beta = min(0.5 + 20 * LongSteelRatio, 1)
        Dductf = displ / dyf

        if self.bending == "single":
            alpha = min([max([1, 3 - self.L / self.h]), 1.5])
            if self.P > 0:
                Vp = (self.P * (self.h - cMn) / (2 * self.L)) / 1000
            else:
                Vp = 0
        elif self.bending == "double":
            alpha = min([max([1, 3 - self.L / 2 / self.h]), 1.5])
            if self.P > 0:
                Vp = (self.P * (self.h - cMn) / self.L) / 1000
            else:
                Vp = 0

        Vc = np.zeros(len(Dductf))
        if self.ductility_mode == "uniaxial":
            for i in range(len(Dductf)):
                Vc[i] = alpha * beta * min(max(0.05, 0.37 - 0.04 * Dductf[i]), 0.29) * 0.8 * self.fc_prime ** 0.5 * \
                        self.b * self.h / 1000
        elif self.ductility_mode == "biaxial":
            for i in range(len(Dductf)):
                Vc[i] = alpha * beta * min(max(0.05, 0.33 - 0.04 * Dductf[i]), 0.29) * 0.8 * self.fc_prime ** 0.5 * \
                        self.b * self.h / 1000
        Vcd = 0.862 * Vc
        Vpd = 0.85 * Vp
        V = Vc + Vs + Vp
        Vd = 0.85 * (Vcd + Vsd + Vpd)
        return V, Vd

    def check_failures(self, Force, V, displ, Dduct, mom, curv, CuDu, dy):
        """
        Calculates parameters of interest at failure
        :param Force: ndarray                       Force
        :param V: ndarray                           Shear strength
        :param displ: ndarray                       Total displacement
        :param Dduct: ndarray                       Ductilities (displ/dy)
        :param mom: ndarray                         Bending moment
        :param curv: ndarray                        Curvature
        :param CuDu: ndarray                        Curvature ductilities
        :param dy: float                            Yield displacement
        :return: ndarray                            Criteria of failure and values of parameters associated with failure
        """
        criteria = 1
        if V[-1] < Force[-1]:
            failure = V - Force
            faildispl = interp1d(failure, displ)(0)
            failforce = interp1d(displ, Force)(faildispl)
            failduct = interp1d(displ, Dduct)(faildispl)
            failmom = interp1d(displ, mom)(faildispl)
            failcurv = interp1d(displ, curv)(faildispl)
            failCuDu = interp1d(displ, CuDu)(faildispl)

            if self.bending == "single":
                if faildispl <= 2 * dy:
                    criteria = 2
                elif faildispl < 8 * dy:
                    criteria = 3
                else:
                    criteria = 4
            elif self.bending == "double":
                if faildispl <= 1 * dy:
                    criteria = 2
                elif faildispl < 7 * dy:
                    criteria = 3
                else:
                    criteria = 4
        else:
            faildispl = None
            failforce = None
            failduct = None
            failmom = None
            failcurv = None
            failCuDu = None

        return np.array([criteria, faildispl, failforce, failduct, failmom, failcurv, failCuDu])

    def get_pars_limit_state(self, ecdam, esser, esdam, coverstrain, steelstrain, displ, Dduct, curv, mom, Force, CuDu):
        """
        Gets parameters associated with the limit states of interest
        :param ecdam: float                             Deformation limit state (concrete)
        :param esser: float                             Deformation limit states (steel)
        :param esdam: float                             Deformation limit states (steel)
        :param coverstrain: ndarray                     Concrete cover strains
        :param steelstrain: ndarray                     Steel strains
        :param displ: ndarray                           Total displacements
        :param Dduct: ndarray                           Ductilities
        :param curv: ndarray                            Curvatures
        :param mom: ndarray                             Bending moments
        :param Force: ndarray                           Forces
        :param CuDu: ndarray                            Curvature ductilities
        :return: ndarray
        """
        displdam = 0
        displser = 0
        Dductdam = 0
        Dductser = 0
        curvdam = 0
        curvser = 0
        CuDudam = 0
        CuDuser = 0
        coverstraindam = 0
        coverstrainser = 0
        steelstraindam = 0
        steelstrainser = 0
        momdam = 0
        momser = 0
        Forcedam = 0
        Forceser = 0

        if max(coverstrain) > self.ECSER or max(abs(steelstrain)) > abs(esser):
            if max(coverstrain) > ecdam or max(abs(steelstrain)) > abs(esdam):
                displdamc = interp1d(coverstrain, displ)(ecdam)
                displdams = interp1d(steelstrain, displ)(esdam)
                displdam = min(displdamc, displdams)
                Dductdam = interp1d(displ, Dduct)(displdam)
                curvdam = interp1d(displ, curv)(displdam)
                coverstraindam = interp1d(displ, coverstrain)(displdam)
                steelstraindam = interp1d(displ, steelstrain)(displdam)
                momdam = interp1d(displ, mom)(displdam)
                Forcedam = interp1d(displ, Force)(displdam)

            displserc = interp1d(coverstrain, displ)(self.ECSER)
            displsers = interp1d(steelstrain, displ)(esser)
            displser = min(displserc, displsers)
            Dductser = interp1d(displ, Dduct)(displser)
            curvser = interp1d(displ, curv)(displser)
            CuDuser = interp1d(displ, CuDu)(displser)
            coverstrainser = interp1d(displ, coverstrain)(displser)
            steelstrainser = interp1d(displ, steelstrain)(displser)
            momser = interp1d(displ, mom)(displser)
            Forceser = interp1d(displ, Force)(displser)

        outputlimit = np.array([coverstrainser, steelstrainser, momser, Forceser, curvser, CuDuser, displser, Dductser,
                                coverstraindam, steelstraindam, momdam, Forcedam, curvdam, CuDudam, displdam, Dductdam,
                                max(coverstrain), min(steelstrain), mom[-1], Force[-1], max(curv), max(CuDu),
                                max(displ), max(Dduct)])
        return outputlimit

    def master(self, Ast):

        MLR = self.get_MLR(Ast)
        eccr = self.get_cracking()
        wi = self.get_wi(MLR)

        # Core height and width
        Hcore = h - 2 * self.clb + self.Dh
        Bcore = b - 2 * self.clb + self.Dh
        # distance to the core
        dcore = self.clb - self.Dh * 0.5
        # thickness of concrete layers
        tcl = h / self.NCL
        # border distance concrete layer
        yl = tcl * np.arange(1, self.NCL + 1, 1)

        esser = -self.ESSER
        esdam = -self.ESDAM

        # Unconfined
        if self.unconfined == "mu":
            ecun, fcun = self.manderun(self.Ec, self.fc_prime, self.eco, self.espall, self.DELS)
        elif self.unconfined == "mc":
            ecun, fcun = self.manderconf(self.Ec, Ast, self.Dh, self.clb, self.s, self.fc_prime, self.fy, self.eco,
                                         self.esm, 'rectangular', self.h, self.b, self.ncx, self.ncy, wi, self.DELS)
        elif self.unconfined == "mclw":
            ecun, fcun = self.manderconflw(self.Ec, Ast, self.Dh, self.clb, self.s, self.fc_prime, self.fy, self.eco,
                                           self.esm, self.espall, 'rectangular', 0, 0, 0, 0, 0, self.DELS)
        else:
            raise ValueError("[EXCEPTION] Wrong unconfined material model...")

        # Confined
        if self.confined == "mu":
            ec, fc = self.manderun(self.Ec, self.fc_prime, self.eco, self.espall, self.DELS)
        elif self.confined == "mc":
            ec, fc = self.manderconf(self.Ec, Ast, self.Dh, self.clb, self.s, self.fc_prime, self.fy, self.eco,
                                     self.esm, 'rectangular', self.h, self.b, self.ncx, self.ncy, wi, self.DELS)
        elif self.confined == "mclw":
            ec, fc = self.manderconflw(self.Ec, Ast, self.Dh, self.clb, self.s, self.fc_prime, self.fy, self.eco,
                                       self.esm, self.espall, 'rectangular', self.h, self.b, self.ncx, self.ncy, wi,
                                       self.DELS)
        else:
            raise ValueError("[EXCEPTION] Wrong confined material model...")

        # Rebars
        if self.rebar == "ks":
            es, fs = self.steelking(self.Es, self.fy, self.fsu, self.esh, self.esu, self.DELS)
        elif self.rebar == "ra":
            es, fs = self.raynor(self.Es, self.fy, self.fsu, self.esh, self.esu, self.DELS, self.C1, self.Ey)
        else:
            raise ValueError("[EXCEPTION] Wrong rebar material model...")

        # maximum strain confined concrete
        ecu = ec[-1]
        # ultimate strain predicted by the original mander model
        ecumander = ecu / 1.5

        if self.ECDAM == "twth":
            ecdam = ecumander
        else:
            ecdam = self.ECDAM

        # vector with strains of confined concrete
        ec = np.hstack((-1e10, ec, np.array(ec[-1] + self.DELS), 1e10))
        # stresses of confined concrete
        fc = np.hstack((0, fc, np.array([0, 0])))

        # strains of unconfined concrete
        ecun = np.hstack((-1e10, ecun, np.array(ecun[-1] + self.DELS), 1e10))
        # stresses of unconfined concrete
        fcun = np.hstack((0, fcun, np.array([0, 0])))

        # maximum steel strain
        esu = es[-1]
        # strains of the steel
        es = np.hstack((es, np.array(es[-1] + self.DELS), 1e10))
        # stresses of steel
        fs = np.hstack((fs, np.array([0, 0])))

        esaux = np.zeros(len(es))
        fsaux = np.zeros(len(fs))
        for i in range(len(es)):
            esaux[i] = es[-1 - i]
            fsaux[i] = fs[-1 - i]

        # strains of steel
        es = np.hstack((-esaux, es[1:]))
        fs = np.hstack((-fsaux, fs[1:]))

        if self.pflag:
            # todo, add plots
            pass

        # --- CONCRETE LAYERS
        # add layers tp consider unconfined concrete
        yl = np.sort(np.hstack((yl, dcore, h-dcore)))
        k = 0
        yaux = np.array([])
        for i in range(len(yl) - 1):
            if yl[i] != yl[i+1]:
                yaux = np.append(yaux, yl[i])
                k = k + 1

        yl = np.hstack([yaux, yl[-1]])
        yc = yl - dcore
        # confined concrete layers
        yc = np.hstack((yc[(yc>0) & (yc < Hcore)], Hcore))

        # total area of each layer
        Atc = np.hstack((yl[0], np.diff(yl)))*b

        # total area of each confined layer
        Atcc = np.hstack((yc[0], np.diff(yc)))*Bcore

        k = 0
        conclay = np.zeros((len(yl), 2))
        for i in range(len(yl)):
            if yl[i] <= dcore or yl[i] > h-dcore:
                conclay[i, :] = np.hstack((Atc[i], 0))
            if dcore < yl[i] <= h-dcore:
                conclay[i, :] = np.hstack((Atc[i] - Atcc[k], Atcc[k]))
                k = k + 1

        # [center layer|A uncon|A conf|d top layer]
        conclay = np.vstack((np.hstack((yl[0]/2, 0.5*(yl[0:-1] + yl[1:len(yl)]))), conclay.T, yl))

        # --- REBARS
        distld = []
        Asbs = []
        diam = []
        for jj in range(len(MLR)):
            distld2 = MLR[jj][0]*np.ones(int(MLR[jj][1]))
            distld = np.hstack((distld, distld2))
            Asbs2 = 0.25*np.pi*(MLR[jj][2]**2)*np.ones(int(MLR[jj][1]))
            Asbs = np.hstack((Asbs, Asbs2))
            diam2 = MLR[jj][2]*np.ones(int(MLR[jj][1]))
            diam = np.hstack((diam, diam2))

        temp_stack = np.vstack((distld, Asbs, diam)).T
        auxqp = temp_stack[temp_stack[:, 0].argsort()]
        # y coordinate of each bar
        distld = auxqp[:, 0]
        # area of each bar
        Asbs = auxqp[:, 1]
        # diameter of each bar
        diam = auxqp[:, 2]

        # --- CORRECTED AREAS
        err = 0
        for i in range(np.size(distld[0])):
            aux = (yl > distld[i]).nonzero()[0]
            conclay[2, aux[0]] = conclay[2, aux[0]] - Asbs[i]
            if conclay[2, aux[0]] < 0:
                err = err + 1

        if err > 0:
            if self.dflag:
                print("[WARNING] Decrease number of layers")

        # Deformations in the top concrete
        ndisp, disp = self.get_disp(ecu, ecun, fcun, ec, fc, es, fs, yl, conclay)

        # --- Iterative process to find the moment-curvature relation
        curv, mom, ejen, DF, vniter, coverstrain, corestrain, steelstrain = \
            self.iterate_for_m_phi(disp, conclay, distld, Asbs, ecun, fcun, ec, fc, es, fs, dcore, ecu, esu)

        Agross = self.h * self.b
        AsLong = Ast
        LongSteelRatio = Ast / Agross
        TransvSteelRatioX = ncx * 0.25 * np.pi * self.Dh ** 2 / self.s / Hcore
        TransvSteelRatioY = ncy * 0.25 * np.pi * self.Dh ** 2 / self.s / Bcore
        TransvSteelRatioAverage = (TransvSteelRatioX + TransvSteelRatioY) * 0.5
        AxialRatio = self.P / self.fc_prime / Agross

        Mn = interp1d(coverstrain, mom)(0.004)
        esaux = interp1d(mom, steelstrain)(Mn)

        if esaux < -0.015:
            Mn = interp1d(steelstrain, mom)(-0.015)
        cMn = interp1d(mom, ejen)(Mn)

        # curvature for first yield
        fycurv = interp1d(steelstrain, curv)(-self.fy / self.Es)
        # moment for first yield
        fyM = interp1d(curv, mom)(fycurv)

        eqcurv = max(Mn / fyM * fycurv, fycurv)

        curvbilin = np.array([0, eqcurv, curv[-1]])
        mombilin = np.array([0, Mn, mom[-1]])

        curvature_ductility = curv[-1] / eqcurv
        marker1 = round(curv[-1] / eqcurv, 1)

        if self.pflag:
            # todo add plots
            pass

        Dbl = max(MLR[:, 2])
        Lsp = np.zeros(len(steelstrain))
        for j in range(len(steelstrain)):
            ffss = -steelstrain[j]*self.Es
            if ffss > self.fy:
                ffss = self.fy
            # strain penetration length
            Lsp[j] = self.kLsp*ffss*Dbl

        kkk = min(0.2*(self.fsu/self.fy - 1), 0.08)

        if self.bending == "single":
            # plastic hinge length
            Lp = max(kkk*L + self.kLsp*self.fy*Dbl, 2*self.kLsp*self.fy*Dbl)
            LBE = L
        elif self.bending == "double":
            # plastic hinge length
            Lp = max(kkk*L/2 + self.kLsp*self.fy*Dbl, 2*self.kLsp*self.fy*Dbl)
            LBE = L/2
        else:
            raise ValueError("[EXCEPTION] Bending should be specified as single or dobule")

        # Buckling models
        CuDu, bucritMK, bucritBE, failCuDuMK, failCuDuBE = self.get_buckling(curv, fycurv, eqcurv, curvature_ductility,
                                                                             steelstrain, diam, TransvSteelRatioAverage,
                                                                             Lp, LBE)

        # Flexure deflection
        displf = np.zeros(len(curv))
        if self.bending == "single":
            for i in range(len(curv)):
                if coverstrain[i] < eccr:
                    displf[i] = curv[i] * ((self.L / 1000) ** 2) / 3
                elif eccr <= coverstrain[i] < fycurv:
                    displf[i] = curv[i] * (((self.L + Lsp[i]) / 1000) ** 2) / 3
                elif curv[i] >= fycurv:
                    displf[i] = (curv[i] - fycurv * (mom[i] / fyM)) * (Lp / 1000) * ((self.L + Lsp[i] - 0.5 * Lp)
                                                                                     / 1000) + \
                                (fycurv * (((self.L + Lsp[i]) / 1000) ** 2) / 3) * mom[i] / fyM
            Force = mom / (self.L / 1000)

        elif self.bending == "double":
            for i in range(len(curv)):
                if coverstrain[i] < eccr:
                    displf[i] = curv[i] * ((self.L / 1000) ** 2) / 6
                elif eccr <= coverstrain[i] < fycurv:
                    displf[i] = curv[i] * (((self.L + 2 * Lsp[i]) / 1000) ** 2) / 6
                elif curv[i] >= fycurv:
                    displf[i] = (curv[i] - fycurv * (mom[i] / fyM)) * (Lp / 1000) * (
                                (self.L + 2 * (Lsp[i] - 0.5 * Lp)) / 1000) + \
                                (fycurv * (((L + 2 * Lsp[i]) / 1000) ** 2) / 6) * mom[i] / fyM
            Force = 2 * mom / (self.L / 1000)

        else:
            raise ValueError("[EXCEPTION] Bending should be specified as single or double")

        # Shear deflection
        G = 0.43 * self.Ec
        As = 5 / 6 * Agross
        Ig = self.b * self.h ** 3 / 12
        Ieff = (Mn * 1000 / (self.Ec * 10 ** 6 * eqcurv)) * 10 ** 12

        beta = min(0.5 + 20 * LongSteelRatio, 1)

        if self.bending == "single":
            alpha = min(max(1, 3 - self.L / self.h), 1.5)
        elif self.bending == "double":
            alpha = min(max(1, 3 - self.L / (2 * self.h)), 1.5)

        Vc1 = 0.29 * alpha * beta * 0.8 * (self.fc_prime ** 0.5) * Agross / 1000

        kscr = (0.25 * TransvSteelRatioY * self.Es * self.b / 1000 * ((self.h - dcore) / 1000) / (
                    0.25 + 10 * TransvSteelRatioY)) * 1000

        if self.bending == "single":
            ksg = G * As / self.L / 1000
            kscr = kscr / self.L
            forcebilin = mombilin / (self.L / 1000)
        elif self.bending == "double":
            ksg = G * As / (self.L / 2) / 1000
            kscr = kscr / (self.L / 2)
            forcebilin = 2 * mombilin / (self.L / 1000)

        kseff = ksg * Ieff / Ig
        aux = Vc1 / kseff / 1000
        aux2 = 0
        momaux = mom
        displsh = np.zeros(len(curv))
        for i in range(len(curv)):
            if momaux[i] <= Mn and Force[i] < Vc1:
                displsh[i] = (Force[i] / kseff) / 1000
            if momaux[i] <= Mn and Force[i] >= Vc1:
                displsh[i] = ((Force[i] - Vc1) / kscr) / 1000 + aux
            if momaux[i] > Mn:
                momaux = 4 * momaux
                aux3 = i - aux2
                aux2 = aux2 + 1
                displsh[i] = (displf[i] / displf[i - 1]) * displsh[i - 1]

        # Total displacement of flexural and shear contributions
        displ = displsh + displf

        # --- BILINEAR APPROXIMATION
        dy1 = interp1d(curv, displ)(fycurv)
        dy = Mn/fyM*dy1
        du = displ[-1]
        displbilin = np.array([0, dy, du])
        Dduct = displ/dy
        DisplDuct = max(Dduct)

        dylf = interp1d(curv, displf)(fycurv)
        dyf = Mn/fyM*dylf
        
        # Shear strength
        V, Vd = self.get_shear_strength(LongSteelRatio, cMn, displ, dyf, alpha)

        # Checking for failures
        failures = self.check_failures(Force, V, displ, Dduct, mom, curv, CuDu, dy)
        criteria = failures[0]

        # Equivalent I for NLTHA
        Ieq = (Mn / (eqcurv * self.Ec)) / 1000
        Bi = 1 / (((mombilin[1]) / (curvbilin[1])) / ((mombilin[2] - mombilin[1]) / (curvbilin[2] - curvbilin[1])))

        # Limit states
        outputlimit = self.get_pars_limit_state(ecdam, esser, esdam, coverstrain, steelstrain, displ, Dduct, curv, mom,
                                                Force, CuDu)
        displdam = outputlimit[14]
        displser = outputlimit[6]

        if self.pflag:
            # todo, add plots
            pass

        dummy1 = np.linspace(0, round(max(displ) * 1000), round(max(displ) * 1000) + 1)
        dummy2 = interp1d(displ * 1000, Force)(dummy1)

        DV = np.hstack((dummy1, dummy2))

        pointsdam = (displ <= displdam).nonzero()[0]
        pointsser = (displ <= displser).nonzero()[0]

        if criteria != 1 and (bucritMK == 1 and bucritBE == 0):
            buckldispl = interp1d(CuDu, displ)(failCuDuMK)
            bucklforce = interp1d(CuDu, Force)(failCuDuMK)

        if criteria != 1 and (bucritMK == 1 and bucritBE == 1):
            buckldispl = interp1d(CuDu, displ)(failCuDuMK)
            bucklforce = interp1d(CuDu, Force)(failCuDuMK)
            buckldisplBE = interp1d(CuDu, displ)(failCuDuMK)
            bucklforceBE = interp1d(CuDu, Force)(failCuDuMK)

        if criteria != 1 and (bucritMK == 0 and bucritBE == 1):
            buckldisplBE = interp1d(CuDu, displ)(failCuDuBE)
            bucklforceBE = interp1d(CuDu, Force)(failCuDuBE)

        if criteria != 1 and (bucritMK == 0 and bucritBE == 0):
            pass

        if criteria == 1 and (bucritMK == 1 and bucritBE == 0):
            buckldispl = interp1d(CuDu, displ)(failCuDuMK)
            bucklforce = interp1d(CuDu, Force)(failCuDuMK)

        if criteria == 1 and (bucritMK == 1 and bucritBE == 1):
            buckldispl = interp1d(CuDu, displ)(failCuDuMK)
            bucklforce = interp1d(CuDu, Force)(failCuDuMK)
            buckldisplBE = interp1d(CuDu, displ)(failCuDuMK)
            bucklforceBE = interp1d(CuDu, Force)(failCuDuMK)

        if criteria == 1 and (bucritMK == 0 and bucritBE == 1):
            buckldisplBE = interp1d(CuDu, displ)(failCuDuBE)
            bucklforceBE = interp1d(CuDu, Force)(failCuDuBE)

        if criteria == 1 and (bucritMK == 0 and bucritBE == 0):
            pass

        # --- OUTPUTS
        output = np.array([coverstrain, corestrain, ejen, steelstrain, mom, curv, Force, displsh, displf, displ, V, Vd])
        outputbilin = np.array([curvbilin, mombilin, displbilin, forcebilin])

        Acore = Hcore * Bcore
        # Compression force for yielding
        PCY = interp1d(ec, fc)(self.fy / self.Es) * (Acore - AsLong) + interp1d(ecun, fcun)(self.fy / self.Es) * (
                Agross - Acore) + AsLong * self.fy
        # axial force for yielding
        PTY = AsLong * self.fy

        # todo, print out results

        # --- Moment axial force interaction
        control = min(self.CSID, self.SSID)

        # compression force for first yielding
        PCid = interp1d(ec, fc)(control) * (Acore - AsLong) + interp1d(ecun, fcun)(control) * (
                    Agross - Acore) + AsLong * interp1d(es, fs)(control)
        # axial force for first yielding
        PTid = AsLong * interp1d(es, fs)(control)

        # Vector with axial loads for interaction diagram
        PP = np.hstack((np.arange(-0.9 * PTid, 0 + 0.3 * PTid, 0.3 * PTid),
                        np.arange(0.05 * self.fc_prime * Agross, 0.6 * PCid, 0.05 * self.fc_prime * Agross),
                        0.7 * PCid, 0.8 * PCid, 0.9 * PCid))

        nPP = len(PP)
        Mni = np.zeros(nPP)
        eci = np.zeros(nPP)
        esi = np.zeros(nPP)
        mess = np.zeros(nPP)
        for i in range(nPP):
            message = 0
            # todo, combine with get_disp function
            if ecu <= 0.0018:
                disp = np.arange(0.0001, 20 * ecu + 0.0001, 0.0001)
            elif 0.0018 < ecu <= 0.0025:
                disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 20 * ecu + 0.0002, 0.0002)))
            elif 0.0025 < ecu <= 0.006:
                disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                                  np.arange(0.0025, 20 * ecu + 0.0005, 0.0005)))
            elif 0.006 < ecu <= 0.012:
                disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                                  np.arange(0.0025, 0.0055, 0.0005), np.arange(0.006, 20 * ecu + 0.001, 0.001)))
            elif ecu > 0.012:
                disp = np.hstack((np.arange(0.0001, 0.0017, 0.0001), np.arange(0.0018, 0.0022, 0.0002),
                                  np.arange(0.0025, 0.0055, 0.0005), np.arange(0.006, 0.011, 0.001),
                                  np.arange(0.012, 20 * ecu + 0.002, 0.002)))

            ndisp = len(disp)

            if PP[i] > 0:
                for j in range(ndisp):
                    compch = sum(interp1d(ecun, fcun)(disp[0] * np.ones(len(yl))) * conclay[1, :]) + \
                             sum(interp1d(ec, fc)(disp[0] * np.ones(len(yl))) * conclay[2, :]) + \
                             Ast * interp1d(es, fs)(disp[0])

                    if compch < PP[i]:
                        disp = disp[1:]

            ndisp = len(disp)

            # curvature
            curv = np.array([0])
            # moments
            mom = np.array([0])
            # neutral axis
            ejen = np.array([0])
            # forcce equilibrium
            DF = np.array([0])
            # iterations
            vniter = np.array([0])

            coverstrain = np.array([0])
            corestrain = np.array([0])
            steelstrain = np.array([0])

            tol = self.TOL * self.h * self.b * self.fc_prime
            x = np.array([self.h / 2])

            for k in range(ndisp):
                F = 10 * tol
                niter = -1
                while abs(F) > tol:
                    niter += 1

                    if x[niter] <= h:
                        eec = (disp[k] / x[niter]) * (conclay[0, :] - (self.h - x[niter]))
                        ees = (disp[k] / x[niter]) * (distld - (self.h - x[niter]))
                    elif x[niter] > h:
                        eec = (disp[k] / x[niter]) * (x[niter] - self.h + conclay[0, :])
                        ees = (disp[k] / x[niter]) * (x[niter] - self.h + distld)

                    eec[eec < min(ecun)] = min(ecun)
                    eec[eec > max(ecun)] = max(ecun)
                    fcunconf = interp1d(ecun, fcun)(eec)
                    eec[eec < min(ec)] = min(ec)
                    eec[eec > max(ec)] = max(ec)
                    fcconf = interp1d(ec, fc)(eec)
                    ees[ees < min(es)] = min(es)
                    ees[ees > max(es)] = max(es)
                    fsteel = interp1d(es, fs)(ees)
                    FUNCON = fcunconf * conclay[1, :]
                    FCONF = fcconf * conclay[2, :]
                    FST = Asbs * fsteel
                    F = sum(FUNCON) + sum(FCONF) + sum(FST) - PP[i]
                    if F > 0:
                        x = np.append(x, x[niter] - 0.05 * x[niter])
                    elif F < 0:
                        x = np.append(x, x[niter] + 0.05 * x[niter])
                    if niter > self.ITERMAX:
                        message = 3
                        break
                cores = (disp[k] / x[niter]) * abs(x[niter] - dcore)
                TF = (self.confined == self.unconfined)
                if not TF:
                    if cores >= ecu:
                        message = 1
                        break
                if TF:
                    if disp[k] >= ecu:
                        message = 1
                        break

                if abs(ees[0]) > esu:
                    message = 2
                    break

                ejen = np.append(ejen, x[niter])
                DF = np.append(DF, F)
                vniter = np.append(vniter, niter)
                mom = np.append(mom, (sum(FUNCON * conclay[0, :]) + sum(FCONF * conclay[0, :]) + sum(FST * distld) - PP[
                    i] * h / 2) / 10 ** 6)
                curv = np.append(curv, 1000 * disp[k] / x[niter])
                coverstrain = np.append(coverstrain, disp[k])
                corestrain = np.append(corestrain, cores)
                steelstrain = np.append(steelstrain, ees[0])
                x = np.array([x[niter]])
                if message != 0:
                    break

            try:
                Mni[i] = interp1d(coverstrain, mom)(self.CSID)
            except:
                Mni[i] = np.nan
            try:
                esaux = interp1d(coverstrain, steelstrain)(self.CSID)
            except:
                esaux = np.nan
            # concrete control
            cr = 0
            if abs(esaux) > abs(self.SSID) or Mni[i] == np.nan:
                # steel control
                cr = 1
                Mni[i] = interp1d(steelstrain, mom)(-self.SSID)
            if cr == 0:
                eci[i] = self.CSID
                esi[i] = esaux
            elif cr == 1:
                esi[i] = -self.SSID
                eci[i] = interp1d(steelstrain, coverstrain)(-self.SSID)

            mess[i] = message

        Mni = np.hstack((0, Mni, 0))

        PPn = np.hstack((-PTid, PP, PCid))

        MB = max(Mni)
        PB = PPn[Mni == MB]

        PB13 = 1 / 3 * PB
        MB13 = interp1d(PPn, Mni)(PB13)

        PB23 = 2 / 3 * PB
        MB23 = interp1d(PPn, Mni)(PB23)

        MB0 = interp1d(PPn, Mni)(0)

        PPL = np.hstack((-PTid, 0, PB13, PB23, PB, PCid))
        MnL = np.hstack((0, MB0, MB13, MB23, MB, 0))

        outputi = np.vstack((Mni, PPn / 1000))


if __name__ == '__main__':

    h = 500
    b = 400
    ncx = 4
    ncy = 4
    cover = 25
    Mtarget = 300.
    L = 5000
    Ast = 1200

    mphi = MomentCurvatureRC1(b, h, ncx, ncy, cover, Mtarget, L, P=700, nlayers=2)
    start_time = mphi.get_init_time()
    mphi.master(Ast)
    mphi.get_time(start_time)
