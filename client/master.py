"""
Runs the integrated seismic risk and economic loss driven framework
"""
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
from client.input import Input
from calculations.hazard import Hazard
from calculations.lossCurve import LossCurve
from calculations.spectra import Spectra
from calculations.designLimits import DesignLimits
from calculations.transformations import Transformations
from calculations.periodRange import PeriodRange
from external.crossSection import CrossSection
from external.crossSectionSpace import CrossSectionSpace
from verifications.mafcCheck import MAFCCheck
from postprocessing.action import Action
from external.openseesrun import OpenSeesRun
from external.openseesrun3d import OpenSeesRun3D
from postprocessing.detailing import Detailing
from external.elf import ELF


class Master:
    def __init__(self, mainDirectory, flag3d=False):
        """
        initialize IPBSD
        :param mainDirectory: str                   Main directory of tool
        :param flag3d: bool                         False for "2d", True for "3d"
        """
        self.dir = mainDirectory
        self.flag3d = flag3d
        self.coefs = None                           # Hazard fitting coefficients               dict
        self.hazard_data = None                     # Hazard information                        dict
        self.original_hazard = None                 # Original hazard information               dict
        self.data = None                            # Input data                                dict
        self.g = 9.81                               # Ground motion acceleration in m/s2        float

    def read_input(self, input_file, hazard_file, outputPath):
        """
        reads input data
        :param input_file: str                      Input filename
        :param hazard_file: str                     Hazard filename
        :param outputPath: str                      Outputs path
        :return None:
        """
        self.data = Input(self.flag3d)
        self.data.read_inputs(input_file)
        self.coefs, self.hazard_data, self.original_hazard = self.data.read_hazard(hazard_file, outputPath)

    def get_hazard_pga(self, lambda_target):
        """
        Gets hazard at pga
        :param lambda_target: float                 Target MAFC
        :return: array                              Array of mean annual frequency of exceedance of limit states
        """
        coef = self.coefs['PGA']
        h = Hazard(coef, "PGA", return_period=self.data.TR, beta_al=self.data.beta_al)
        lambda_ls = h.lambdaLS
        # Set MAFE of CLS to target MAFC
        lambda_ls[-1] = lambda_target
        return lambda_ls

    def get_hazard_sa(self, period, hazard):
        """
        gets the hazard at a given period
        :param period: float                        Fundamental period of the structure
        :param hazard: str                          Hazard to use, i.e. True or Fitted
        :return: array, array                       Probabilities and Sa range of the hazard
        """
        if hazard == "Fitted":
            coef = self.coefs['SA({})'.format(str(period))]
            h = Hazard(coef, "SA", beta_al=self.data.beta_al)
            Hs = h.Hs
            Sa = h.Sa_Range
        elif hazard == "True":
            Hs = self.original_hazard[2][int(round(period*10))]
            Sa = self.original_hazard[1][int(round(period*10))]
        else:
            raise ValueError("[EXCEPTION] Wrong hazard type, should be Fitted or True")
        return Hs, Sa

    def get_loss_curve(self, lambda_ls, eal_limit):
        """
        gets the loss curve
        :param lambda_ls: array                     MAF of exceeding performance limit states
        :param eal_limit: float                     EAL limit value
        :return: float                              Computed EAL as the area below the loss curve
        """
        lc = LossCurve(self.data.y, lambda_ls, eal_limit)
        lc.verify_eal()
        return lc.EAL, lc.y_fit, lc.lambda_fit

    def get_spectra(self, lam):
        """
        gets spectra at SLS
        :param lam: float                           MAF of exceeding SLS
        :return: array, array                       Spectral accelerations [g] and spectral displacements [cm] at SLS
                                                    and Period range [s]
        """
        s = Spectra(lam, self.coefs, self.hazard_data['T'], self.original_hazard)
        return s.sa, s.sd, s.T_RANGE

    def get_design_values(self, slfDirectory, replCost=None, eal_corrections=True, perform_scaling=True,
                          edp_profiles=None):
        """
        gets the design values of IDR and PFA from the Storey-Loss-Functions
        :param slfDirectory: str                    Directory of SLFs derived via SLF Generator
        :param replCost: float                      Replacement cost of the entire building
        :param eal_corrections: bool                Perform EAL corrections
        :param perform_scaling: bool                Perform scaling of SLFs to replCost
        :param edp_profiles: list                   0 = pfa profiles, 1 = psd profiles (DEV variable, optional)
                                                    Force tool to use specific EDP profiles
        :return: float, float                       Peak storey drift, (PSD) [-] and Peak floor acceleration, (PFA) [g]
        """
        y_sls = self.data.y[1]
        dl = DesignLimits(slfDirectory, y_sls, self.data.nst, self.flag3d, replCost, eal_corrections, perform_scaling,
                          edp_profiles)
        slfsCache = dl.SLFsCache
        if eal_corrections:
            self.data.y[1] = dl.y

        return dl.theta_max, dl.a_max, slfsCache, dl.contributions

    def perform_transformations(self, th, a):
        """
        performs design to spectral value transformations
        :param th: float                            Peak storey drift, PSD, [-]
        :param a: float                             Peak floor acceleration, PFA, [g]
        :return: dict, float, float                 DDBD at SLS, Design spectral displacement and acceleration
        """
        delta = np.zeros(th.shape)
        alpha = np.zeros(a.shape)
        tables = {}
        for i in range(th.shape[0]):
            t = Transformations(self.data, th[i], a[i])
            table, phi, deltas = t.table_generator()
            g, m = t.get_modal_parameters(phi)
            delta[i], alpha[i] = t.get_design_values(deltas)
            tag = "x" if i == 0 else "y"
            tables[tag] = table

        return tables, delta, alpha

    def get_period_range(self, d, a, sd, sa):
        """
        gets the feasible initial period range
        :param d: float                             Design spectral displacement in m
        :param a: float                             Design spectral acceleration in g
        :param sd: array                            Spectral displacements at SLS in m
        :param sa: array                            Spectral accelerations at SLS in g
        :return: float, float                       Allowable lower and upper periods in s
        """
        pr = PeriodRange(d, a, sd, sa)
        new_sa_sls, new_sd_sls = pr.get_new_spectra()
        t_lower = pr.get_T_lower(new_sa_sls, new_sd_sls)
        t_upper = pr.get_T_upper(new_sa_sls, new_sd_sls)
        pr.period_range_verification(t_lower, t_upper)
        return t_lower, t_upper

    def get_all_section_combinations(self, period_limits, fstiff=0.5, solution_x=None, solution_y=None, data=None,
                                     cache_dir=None):
        """
        gets all section combinations satisfying period bound range
        :param period_limits: list                  Lower period limit and Upper period limit
        :param fstiff: float                        Stiffness reduction factor
        :param solution_x: Series or int            Solution to run analysis instead (for iterations, dir1
        :param solution_y: Series or int            Solution to run analysis instead (for iterations, dir2
        :param data: object                         IPBSD initial input arguments (object Input)
        :param cache_dir: str                       Directory to export the cache csv solutions of
        :return cs.solutions: DataFrame             Solution combos
        :return opt_sol: DataFrame                  Optimal solution
        :return opt_modes: dict                     Periods and normalized modal shapes of the optimal solution
        """
        def get_system(data, direction):
            if data.configuration == "perimeter":
                n_seismic = 2
                masses = data.masses
            else:
                q_floor = data.i_d['bldg_ch'][0]
                q_roof = data.i_d['bldg_ch'][1]
                n_seismic = 1
                # Considering the critical frame only (with max tributary length)
                if direction == "x":
                    # Spans in the primary direction
                    spans = data.spans_x
                    # Spans in the orthogonal direction
                    spans_ort = np.array(data.spans_y)
                else:
                    spans = data.spans_y
                    spans_ort = np.array(data.spans_x)

                # Get the tributary lengths for the mass
                dist = np.convolve(spans_ort, np.ones(2), "valid") / 2
                tributary = max(dist)
                # Get the mass per storey
                masses = np.zeros((data.nst, ))
                for st in range(data.nst):
                    if st == data.nst - 1:
                        masses[st] = q_roof * sum(spans) * tributary / 9.81
                    else:
                        masses[st] = q_floor * sum(spans) * tributary / 9.81

            return n_seismic, masses

        if self.data.configuration == "perimeter" or not self.flag3d:
            # Get number of seismic frames and lumped masses along the height
            n_seismic, masses = get_system(self.data, "x")
            if solution_x is None:
                t_lower, t_upper = period_limits["1"]
                cs = CrossSection(self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x,
                                  self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                  cache_dir=cache_dir/"solution_cache_x.csv")
                opt_sol, opt_modes = cs.find_optimal_solution()
                results_x = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

            elif solution_x is not None and data is None:
                t_lower, t_upper = period_limits["1"]
                cs = CrossSection(self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x,
                                  self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                  cache_dir=cache_dir/"solution_cache_x.csv")
                opt_sol, opt_modes = cs.find_optimal_solution(solution_x)
                results_x = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

            else:
                t_lower, t_upper = period_limits["1"]
                self.data = data
                n_seismic, masses = get_system(self.data, "x")
                cs = CrossSection(self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x,
                                  self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                  iteration=True)
                opt_sol, opt_modes = cs.find_optimal_solution(solution_x)
                results_x = {"opt_sol": opt_sol, "opt_modes": opt_modes}

            n_seismic, masses = get_system(self.data, "y")
            if self.flag3d:
                # Optimal solution in primary direction (dir1 or x)
                # Essentially the solution in Y direction will be selected by fixing the external columns
                opt_sol_x = results_x["opt_sol"]
                if solution_y is None:
                    t_lower, t_upper = period_limits["2"]
                    cs = CrossSection(self.data.nst, len(self.data.spans_y), self.data.fy, self.data.fc,
                                      self.data.spans_y, self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                      cache_dir=cache_dir/"solution_cache_y.csv", solution_perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution()
                    results_y = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

                elif solution_y is not None and data is None:
                    t_lower, t_upper = period_limits["2"]
                    cs = CrossSection(self.data.nst, len(self.data.spans_y), self.data.fy, self.data.fc,
                                      self.data.spans_y, self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                      cache_dir=cache_dir/"solution_cache_y.csv", solution_perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution(solution_y)
                    results_y = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

                else:
                    t_lower, t_upper = period_limits["2"]
                    self.data = data
                    n_seismic, masses = get_system(self.data, "y")
                    cs = CrossSection(self.data.nst, len(self.data.spans_y), self.data.fy, self.data.fc,
                                      self.data.spans_y, self.data.heights, n_seismic, masses, fstiff, t_lower, t_upper,
                                      iteration=True, solution_perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution(solution_y)
                    results_y = {"opt_sol": opt_sol, "opt_modes": opt_modes}

                return results_x, results_y
            else:
                return results_x
        else:
            # Space systems (3D only)
            if solution_x is None and data is not None:
                cs = CrossSectionSpace(data, period_limits, fstiff)
                cs.read_solutions(cache_dir=cache_dir / "solution_cache_space.csv")
                opt_sol_raw, opt_modes = cs.find_optimal_solution()
                # Convert optimal solution for usability. Gravity refers to central structural elements.
                # (e.g. transform it into a dictionary with keys: x_seismic, y_seismic, gravity)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}
            elif solution_x is not None and data is None:
                cs = CrossSectionSpace(data, period_limits, fstiff)
                cs.read_solutions(cache_dir=cache_dir / "solution_cache_space.csv")
                opt_sol_raw, opt_modes = cs.find_optimal_solution(solution_x)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}
            else:
                cs = CrossSectionSpace(data, period_limits, fstiff, iteration=True)
                cs.read_solutions()
                opt_sol_raw, opt_modes = cs.find_optimal_solution(solution_x)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}
            return results

    def perform_spo2ida(self, spo_pars):
        """
        run spo2ida and identify the collapse fragility
        :param spo_pars: dict                       Dictionary containing spo assumptions
        :return: dict                               Dictionary containing SPO2IDA results
        """
        R16, R50, R84, idacm, idacr, spom, spor = self.data.read_spo(spo_pars, run_spo=True)
        spo2ida_data = {'R16': R16, 'R50': R50, 'R84': R84, 'idacm': idacm, 'idacr': idacr, 'spom': spom, 'spor': spor}
        return spo2ida_data

    def verify_mafc(self, period, spo2ida, g, mafc_target, omega, hazard="True"):
        """
        optimizes for a target mafc
        :param period: float                        Fundamental period of the structure
        :param spo2ida: dict                        Dictionary containing SPO2IDA results
        :param g: float                             First mode participation factor
        :param mafc_target: float                   Target MAFC
        :param omega: float                         Overstrength factor
        :param hazard: str                          Hazard to use, i.e. True or Fitted
        :return: float, float                       Spectral acceleration [g] and displacement [m] at yield
        """
        r = [spo2ida['R16'], spo2ida['R50'], spo2ida['R84']]
        Hs, sa_hazard = self.get_hazard_sa(period, hazard)
        m = MAFCCheck(r, mafc_target, g, Hs, sa_hazard, omega, hazard)
        fsolve(m.objective, x0=np.array([0.02]), factor=0.1)
        # Yield displacement for ESDOF
        dy = float(m.cy) * 9.81 * (period / 2 / np.pi) ** 2
        cy = float(m.cy)
        return cy, dy

    def get_index(self, target, data):
        """
        Gets index of target value
        :param target: float                        Target value
        :param data: list                           Data to look in
        :return: int                                Index of target value
        """
        if np.where(data >= target)[0].size == 0:
            return np.nan
        else:
            return np.where(data >= target)[0][0]

    def get_sa_at_period(self, say, sa, periods, periods_of_interest):
        """
        Generates acceleration based on provided period and target yield acceleration
        :param say: float                           Spectral acceleration at yield
        :param sa: list                             List of interest to generate from
        :param periods: list                        List used to get index of target variable
        :param periods_of_interest: list            Target variables
        :return: list                               Target values generated from list of interest
        """
        se = np.array([])
        scaling_factor = say / sa[self.get_index(periods_of_interest[0], periods)]
        for p in periods_of_interest:
            p = round(p, 2)
            se = np.append(se, sa[self.get_index(p, periods)]*scaling_factor)
        return se

    def get_correlation_matrix(self, periods, num_modes, damping=.05):
        """
        Gets correlation matrix
        :param periods: list                        Periods
        :param num_modes: int                       Number of modes to consider
        :param damping: float                       Ratio of critical damping
        :return: ndarray                            Correlation matrix
        """
        corr = np.zeros([num_modes, num_modes])
        for j in range(num_modes):
            for i in range(num_modes):
                corr[i, j] = (8*damping**2*(periods[j]/periods[i])**(3/2)) / \
                              ((1+(periods[j]/periods[i]))*((1-(periods[j]/periods[i]))**2 +
                                                            4*damping**2*(periods[j]/periods[i])))
        return corr

    def perform_cqc(self, corr, demands):
        """
        Performs complete quadratic combination (CQC)
        :param corr: ndarray                        Correlation matrix
        :param demands: dict                        Demands on structural elements
        :return: ndarray                            Critical response demand on structural elements
        """
        tempNeg = None
        tempPos = None
        temp = None

        num_modes = len(corr)
        response = {}
        b = np.zeros((self.data.nst, self.data.n_bays))
        c = np.zeros((self.data.nst, self.data.n_bays + 1))

        # Initialize results
        results = {"Beams": {"M": {"Pos": b.copy(), "Neg": b.copy()}, "N": b.copy(), "V": b.copy()},
                   "Columns": {"M": c.copy(), "N": c.copy(), "V": c.copy()}}

        # For each element type (Beams, Columns)
        for eleType in demands["Mode1"].keys():
            for dem in demands["Mode1"][eleType].keys():
                if eleType == "Beams" and dem == "M":
                    tempPos = np.zeros(demands["Mode1"][eleType][dem]["Pos"].shape)
                    tempNeg = np.zeros(demands["Mode1"][eleType][dem]["Neg"].shape)
                else:
                    temp = np.zeros(demands["Mode1"][eleType][dem].shape)
                # Star the combination
                for i in range(num_modes):
                    for j in range(num_modes):
                        if eleType == "Beams" and dem == "M":
                            tempPos = tempPos + corr[i][j]*demands[f"Mode{i+1}"][eleType][dem]["Pos"] * \
                                      demands[f"Mode{j+1}"][eleType][dem]["Pos"]
                            tempNeg = tempNeg + corr[i][j]*demands[f"Mode{i+1}"][eleType][dem]["Neg"] * \
                                      demands[f"Mode{j+1}"][eleType][dem]["Neg"]
                        else:
                            temp = temp + corr[i][j]*demands[f"Mode{i+1}"][eleType][dem] * \
                                   demands[f"Mode{j+1}"][eleType][dem]

                if eleType == "Beams" and dem == "M":
                    results[eleType][dem]["Pos"] = np.sqrt(tempPos)
                    results[eleType][dem]["Neg"] = np.sqrt(tempNeg)
                else:
                    results[eleType][dem] = np.sqrt(temp)

        return results

    def get_action(self, solution, cy, df, gravity_loads, analysis, num_modes=None, opt_modes=None,
                   modal_sa=None):
        """
        gets demands on the structure
        :param solution: DataFrame                  Solution containing cross-section and modal properties
        :param cy: float                            Spectral acceleration at yield
        :param df: DataFrame                        Contains information of displacement shape at SLS
        :param analysis: int                        Analysis type
        :param gravity_loads: dict                  Gravity loads as {'roof': *, 'floor': *}
        :param num_modes: int                       Number of modes to consider for SRSS
        :param opt_modes: dict                      Periods and normalized modal shapes of the optimal solution
        :param modal_sa: list                       Spectral acceleration to be used for RMSA
        :return: DataFrame                          Acting forces
        """
        a = Action(solution, self.data.n_seismic, self.data.nst, self.data.masses, cy, df, analysis,
                   gravity_loads, num_modes, opt_modes, modal_sa, self.data.pdelta_loads)
        d = a.forces()
        return d

    def run_muto_approach(self, solution, loads, heights, widths):
        """
        Runs simplified lateral analysis based on Muto's approach
        :return: dict                               Demands on the structural elements
        """
        elf = ELF(solution, loads, heights, widths)
        return elf.response

    def run_analysis(self, analysis, solution, lat_action=None, grav_loads=None, sls=None, yield_sa=None, fstiff=0.5,
                     hinge=None, direction=0, system="perimeter"):
        """
        Runs structural analysis to identify demands on the structural elements
        :param analysis: int                        Analysis type
        :param solution: DataFrame                  Optimal solution
        :param lat_action: list                     Acting lateral loads in kN
        :param grav_loads: list                     Acting gravity loads in kN/m
        :param sls: dict                            Table at SLS, necessary for simplified computations only
        :param yield_sa: float                      Spectral acceleration at yield
        :param fstiff: float                        Stiffness reduction factor
        :param hinge: DataFrame                     Hinge models
        :param direction: bool                      0 for x direction, 1, for y direction
        :param system: str                          System type (perimeter or space), for 3D only
        :return: dict                               Demands on the structural elements
        """
        if analysis == 1:       # redundant, unnecessary, but will be left here as a placeholder for future changes
            if direction == 0:
                nbays = self.data.n_bays
            else:
                nbays = len(self.data.spans_y)

            print("[INITIATE] Starting simplified approximate demand estimation...")
            response = pd.DataFrame({'Mbi': np.zeros(self.data.nst),
                                     'Mci': np.zeros(self.data.nst),
                                     'Mce': np.zeros(self.data.nst)})

            base_shear = yield_sa * solution["Mstar"] * solution["Part Factor"] * self.g
            masses = self.data.masses / self.data.n_seismic
            modes = [sls[str(st+1)]['phi'] for st in range(self.data.nst)]
            # lateral forces
            forces = np.zeros(self.data.nst)
            # shear at each storey level
            shear = np.zeros(self.data.nst)
            for st in range(self.data.nst):
                forces[st] = masses[st] * modes[st] * base_shear / sum(map(lambda x, y: x * y, masses, modes))
            for st in range(self.data.nst):
                shear[st] = sum(fi for fi in forces[st:self.data.nst])

            # Demands on beams and columns in kNm
            # Assuming contraflexure point at 0.6h for the columns
            for st in range(self.data.nst):
                if st != self.data.nst - 1:
                    response['Mbi'][st] = 1/2/nbays * self.data.h[st]/2 * (shear[st] + shear[st+1])
                else:
                    response['Mbi'][st] = 1/2/nbays * self.data.h[st]/2 * shear[st]
                # The following is based on assumption that beam stiffness effects are neglected
                ei_external = solution[f"he{st+1}"]**4
                ei_internal = solution[f"hi{st+1}"]**4
                ei_ratio = ei_internal / ei_external
                ei_total = 2 + ei_ratio*(nbays - 1)
                shear_external = shear[st] / ei_total
                shear_internal = shear_external * ei_ratio
                response['Mci'][st] = 0.6 * self.data.h[st] * shear_internal
                response['Mce'][st] = 0.6 * self.data.h[st] * shear_external
        else:
            if self.flag3d:
                op = OpenSeesRun3D(self.data, solution, fstiff=fstiff, hinge=hinge, direction=direction, system=system)
                response = op.elastic_analysis_3d(analysis, lat_action, grav_loads)

            else:
                op = OpenSeesRun(self.data, solution, fstiff=fstiff, hinge=hinge)
                response = op.elastic_analysis(analysis, lat_action, grav_loads)

        return response

    def ma_analysis(self, solution, hinge, action, fstiff, direction=0):
        """
        Runs modal analysis
        :param solution: DataFrame                  Design solution, cross-section dimensions
        :param hinge: DataFrame                     Idealized plastic hinge model parameters
        :param action: DataFrame                    Gravity loads over PDelta columns
        :param fstiff: float                        Stiffness reduction factor
        :param direction: bool
        :return: list                               Modal periods
        """
        if hinge is None:
            hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

        if self.flag3d:
            ma = OpenSeesRun3D(self.data, solution, fstiff, hinge=hinge, direction=direction,
                               system=self.data.configuration)
        else:
            ma = OpenSeesRun(self.data, solution, fstiff, hinge=hinge)
        ma.create_model()
        ma.define_masses()
        if not self.flag3d:
            ma.pdelta_columns(action)
            num_modes = 1
        else:
            num_modes = self.data.nst if self.data.nst <= 9 else 9
        model_periods, modalShape, gamma, mstar = ma.ma_analysis(num_modes)
        ma.wipe()
        return model_periods, modalShape, gamma, mstar

    def spo_opensees(self, solution, hinge, action, fstiff, modalShape=None, direction=0):
        """
        Runs static pushover analysis and fits an idealized curve to the SPO curve for later use by SPO2IDA
        :param solution: DataFrame                  Design solution, cross-section dimensions
        :param hinge: DataFrame                     Idealized plastic hinge model parameters
        :param action: DataFrame                    Gravity loads over PDelta columns
        :param fstiff: float                        Stiffness reduction factor
        :param modalShape: list                     Modal shape to be used for SPO loads
        :param direction: bool
        :return: dict                               SPO response in terms of top displacement vs base shear
        """
        if self.flag3d:
            system = self.data.configuration
            spo = OpenSeesRun3D(self.data, solution, fstiff, hinge=hinge, direction=direction, system=system)
        else:
            spo = OpenSeesRun(self.data, solution, fstiff, hinge=hinge)
        spo.create_model(gravity=True)
        spo.define_masses()
        if not self.flag3d:
            spo.pdelta_columns(action)
        topDisp, baseShear = spo.spo_analysis(load_pattern=2, mode_shape=modalShape)
        spo.wipe()
        return topDisp, baseShear

    def design_elements(self, demands, sections, modes, dy, ductility_class="DCM", cover=0.03, est_ductilities=True,
                        direction=0, gravity=False):
        """
        Runs M-phi to optimize for reinforcement for each section
        :param demands: DataFrame or dict           Demands identified from a structural analysis (lateral+gravity)
        :param sections: DataFrame                  Solution including section information
        :param modes: dict                          Periods and modal shapes obtained from modal analysis
        :param dy: float                            System yield displacement in m
        :param ductility_class: str                 Ductility class (DCM or DCH, following Eurocode 8 recommendations)
        :param cover: float                         Reinforcement cover in m
        :param est_ductilities: bool                Whether to estimate hardening and fracturing ductilities
        :param direction: bool                      0 for x direction, 1 for y direction
        :param gravity: bool                        Design gravity frames condition
        :return: dict                               Designed element properties from the moment-curvature relationship
        """
        if direction == 0:
            nbays = self.data.n_bays
            spans = self.data.spans_x
        else:
            spans = self.data.spans_y
            nbays = len(spans)

        d = Detailing(demands, self.data.nst, nbays, self.data.fy, self.data.fc, spans, self.data.heights,
                      self.data.n_seismic, self.data.masses, dy, sections, ductility_class=ductility_class,
                      rebar_cover=cover, est_ductilities=est_ductilities, direction=direction)
        if gravity:
            hinge_models, w = d.design_gravity()
            warnMax = d.WARNING_MAX
            warnMin = d.WARNING_MIN
            warnings = {"warnings": w, "warnMax": warnMax, "warnMin": warnMin}

            return hinge_models, warnings
        else:
            data, hinge_models, mu_c, mu_f, warnings = d.design_elements(modes)
            warnMax = d.WARNING_MAX
            warnMin = d.WARNING_MIN

        return data, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings
