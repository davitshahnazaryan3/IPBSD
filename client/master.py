"""
Runs the integrated seismic risk and economic loss driven framework
"""
from pathlib import Path
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
from verifications.periodCheck import PeriodCheck
from verifications.mafcCheck import MAFCCheck
from postprocessing.action import Action
from external.openseesrun import OpenSeesRun
from postprocessing.detailing import Detailing


# todo, add gravity analysis with ELFM
# todo, fix MAFC calculation, MAF to Hs
# todo, add option for MAFC calculation, if H is provided, use it directly, if not use fitted H
# todo, fix, should be cdf not pdf for MAFC calculation
# todo, remove todos on fstiff, fstiff is still necessary for internal elastic section definition
class Master:
    def __init__(self, dir):
        """
        initialize IPBSD
        """
        self.dir = dir
        self.coefs = None
        self.hazard_data = None
        self.original_hazard = None
        self.data = None
        self.REBAR_COVER = 0.03
        self.g = 9.81

    def read_input(self, input_file, hazard_file):
        """
        reads input data
        :param input_file: str                      Input filename
        :param hazard_file: str                     Hazard filename
        :return None:
        """
        self.data = Input()
        self.data.read_inputs(self.dir / "client" / input_file)
        self.coefs, self.hazard_data, self.original_hazard = self.data.read_hazard(self.dir / "external", hazard_file)

    def get_hazard_pga(self, lambda_target):
        """
        gets hazard at pga
        :param lambda_target: float                 Target MAFC
        :return: array, array
        """
        coef = self.coefs['PGA']
        h = Hazard(coef, "PGA", return_period=self.data.TR, beta_al=self.data.beta_al)
        lambda_ls = h.lambdaLS
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
        :return: float                              Computed EAL as the area below the loss curve
        """
        lc = LossCurve(self.data.y, lambda_ls, eal_limit)
        lc.verify_eal()
        return lc.EAL

    def get_spectra(self, lam):
        """
        gets spectra at SLS
        :param lam: float                           MAF of exceeding SLS
        :return: array, array                       Spectral accelerations and spectral displacements at SLS and
                                                    Period range
        """
        s = Spectra(lam, self.coefs, self.hazard_data['T'])
        return s.sa, s.sd, s.T_RANGE

    def get_design_values(self, slf_data):
        """
        gets the design values of IDR and PFA from the Storey-Loss-Functions
        :param slf_data: str                        SLF filename as "*.xlsx"
        :return: float, float                       Peak storey drift [-] and Peak floor acceleration [g]
        """
        slf_filename = self.dir / "client" / slf_data
        dl = DesignLimits(slf_filename, self.data.y)
        return dl.theta_max[0], dl.a_max[0]

    def perform_transformations(self, th, a):
        """
        performs design to spectral value transformations
        :param th: float                            Peak storey drift [-]
        :param a: float                             Peak floor acceleration [g]
        :return: dict, float, float, float
        """
        t = Transformations(self.data, th, a)
        table, phi, deltas = t.table_generator()
        g, m = t.get_modal_parameters(phi)
        delta, alpha = t.get_design_values(deltas)
        return table, g, delta, alpha

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

    def get_all_section_combinations(self, t_lower, t_upper, fstiff=0.5):
        """
        gets all section combinations satisfying period bound range
        :param t_lower: float                       Lower period limit
        :param t_upper: float                       Upper period limit
        :param fstiff: float                        Stiffness reduction factor
        :return cs.solutions: DataFrame             Solution combos
        :return opt_sol: DataFrame                  Optimal solution
        :return opt_modes: dict                     Periods and normalized modal shapes of the optimal solution
        """
        cs = CrossSection(self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x,
                          self.data.h, self.data.n_seismic, self.data.masses, fstiff, t_lower, t_upper)
        opt_sol, opt_modes = cs.find_optimal_solution()
        return cs.solutions, opt_sol, opt_modes

    def perform_spo2ida(self, spo_pars):
        """
        run spo2ida and identify the collapse fragility
        :param spo_pars: dict                       Dictionary containing spo assumptions
        :return: dict                               Dictionary containing SPO2IDA results
        """
        R16, R50, R84, idacm, idacr, spom, spor = self.data.read_spo(spo_pars, run_spo=True)
        spo2ida_data = {'R16': R16, 'R50': R50, 'R84': R84, 'idacm': idacm, 'idacr': idacr, 'spom': spom, 'spor': spor}
        return spo2ida_data

    def verify_period(self, period, tlow, tup):
        """
        verifies if the target period is within a feasible period range
        :param period: float
        :param tlow: float
        :param tup: float
        :return: None
        """
        # todo, period for checking should be rounded to 2 decimals
        # todo, verify that we don't repeat this check from phase 3.1
        p = PeriodCheck(period, tlow, tup)

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
        fsolve(m.objective, x0=np.array([0.05]))
        dy = (m.say*omega)*9.81*(period/2/np.pi)**2
        say = float(m.say)
        return say, dy

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

    def get_sa_at_period(self, sa, periods, periods_of_interest):
        """
        Generates acceleration based on provided and target data
        :param sa: list                             List of interest to generate from
        :param periods: list                        List used to get index of target variable
        :param periods_of_interest: list            Target variables
        :return: list                               Target values generated from list of interest
        """
        se = np.array([])
        for p in periods_of_interest:
            p = round(p, 2)
            se = np.append(se, sa[self.get_index(p, periods)])
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

    def get_action(self, solution, say, df, gravity_loads, analysis, num_modes=None, opt_modes=None, modal_sa=None):
        """
        gets demands on the structure
        :param solution: DataFrame                  Solution containing cross-section and modal properties
        :param say: float                           Spectral acceleration at yield
        :param df: DataFrame                        Contains information of displacement shape at SLS
        :param analysis: int                        Analysis type
        :param gravity_loads: dict                  Gravity loads as {'roof': *, 'floor': *}
        :param num_modes: int                       Number of modes to consider for SRSS
        :param opt_modes: dict                      Periods and normalized modal shapes of the optimal solution
        :param modal_sa: list                       Spectral acceleration to be used for RMSA
        :return: DataFrame                          Acting forces
        """
        # todo, consider phi shape from CrossSection for opt_sol
        a = Action(solution, self.data.n_seismic, self.data.n_bays, self.data.nst, self.data.masses, say, df, analysis,
                   gravity_loads, num_modes, opt_modes, modal_sa)
        d = a.forces()
        return d

    def run_analysis(self, analysis, solution, lat_action=None, grav_loads=None, sls=None, yield_sa=None):
        """
        runs elfm to identify demands on the structural elements
        :param analysis: int                        Analysis type
        :param solution: DataFrame                  Optimal solution
        :param lat_action: list                     Acting lateral loads in kN
        :param grav_loads: list                     Acting gravity loads in kN/m
        :param sls: dict                            Table at SLS, necessary for simplified computations only
        :param yield_sa: float                      Spectral acceleration at yield
        :return: DataFrame or dict                  Demands on the structural elements
        """
        if analysis == 1:
            print("[INITIATE] Starting simplified approximate demand estimation...")
            response = pd.DataFrame({'Mbi': np.zeros(self.data.nst),
                                     'Mci': np.zeros(self.data.nst)})

            base_shear = yield_sa*solution["Mstar"]*solution["Part Factor"]*self.g / self.data.n_seismic
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
                    response['Mbi'][st] = 1/2/self.data.n_bays * self.data.h[st]/2 * (shear[st] + shear[st+1])
                else:
                    response['Mbi'][st] = 1/2/self.data.n_bays * self.data.h[st]/2 * shear[st]
                response['Mci'][st] = 1/2/self.data.n_bays * 0.6 * self.data.h[st] * shear[st]
        else:
            op = OpenSeesRun(self.data, solution, analysis)
            beams, columns = op.create_model()
            if lat_action is not None:
                op.elfm_loads(lat_action)
            if analysis == 3 or analysis == 5:
                if grav_loads is not None:
                    op.gravity_loads(grav_loads, beams)
            op.static_analysis()
            response = op.define_recorders(beams, columns)
        return response

    def design_elements(self, demands, sections, tlower, tupper):
        """
        Runs M-phi to optimize for reinforcement for each section
        :param demands: dict                        Demands identified from a structural analysis (ELFM+gravity)
        :param sections: DataFrame                  Solution including section information
        :return: dict                               Designed element properties from the moment-curvature relationship
        """
        d = Detailing(demands, self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x,
                      self.data.h, self.data.n_seismic, self.data.masses, tlower, tupper, sections)
        data = d.design_elements()
        return data

    def run_ma(self, solution, tlower, tupper, sections):
        """
        runs modal analysis for a single solution
        :param solution: dataframe
        :param tlower: float
        :param tupper: float
        :param sections: dataframe
        :return: float, list
        """
        # todo, verify that we don't repeat calculations from phase 2.3
        fstiff_beam = [sections["Beams"][i][0]["cracked EI"] for i in sections["Beams"]]
        fstiff_col = [sections["Columns"][i][0]["cracked EI"] for i in sections["Columns"]]
        cs = CrossSection(self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x, self.data.h,
                          self.data.n_seismic, self.data.masses, fstiff=1.0, tlower=tlower, tupper=tupper)
        hce, hci, b, h = cs.get_section(solution)

        props = cs.create_props(hce, hci, b, h)
        for i in range(self.data.nst):
            props[2][i] = fstiff_col[i]*props[2][i]
            props[3][i] = fstiff_col[i + self.data.nst] * props[3][i]
            props[5][i] = fstiff_beam[i]*props[5][i]

        period, phi = cs.run_ma(props)
        return period, phi

    def get_system_ductility(self, sections, period, say, details):
        """
        estimates system ductility
        :param sections: dataframe
        :param period: float
        :param say: float
        :param details: dict
        :return: float
        """
        d = Detailing(None, self.data.nst, self.data.n_bays, self.data.fy, self.data.fc, self.data.spans_x, self.data.h,
                      self.data.n_seismic, self.data.masses, 0, 0, sections)
        lp_args = [20, self.data.fy]
        ductility_hard = d.get_hardening_ductility(period, say, details, sections, lp_args)
        print(f"Estimated system hardening ductility: {ductility_hard:.2f}")
        return ductility_hard


if __name__ == "__main__":

    directory = Path.cwd()
    # Phase 1 - ready
    csd = Master(directory)
    csd.read_input("input.csv", "Hazard-LAquila-Soil-C.pkl")
    lambda_ls, pga = csd.get_hazard_pga()
    eal = csd.get_loss_curve(lambda_ls, 0.6)
    slf_file = {0: "client/slf.xlsx"}
    theta_max, a_max = csd.get_design_values(slf_file)
    sls_table, gamma, mstar, delta_d, alpha_d = csd.perform_transformations(theta_max, a_max)
    print("[PHASE] 1 completed!")
    # Phase 2
    sa, sd = csd.get_spectra(lambda_ls[1])
    T_lower, T_upper = csd.get_period_range(delta_d, alpha_d, sd, sa)
    solutions, opt_sol = csd.get_all_section_combinations(T_lower, T_upper, fstiff=0.5)
    print("[PHASE] 2 completed!")
    # Phase 3
    spo = csd.i_d.initial_spo_data(round(float(opt_sol['T']), 1))
    spo2ida_info = csd.perform_spo2ida(spo)
    say, dy = csd.verify_mafc(spo['T'], spo2ida_info, gamma)
    print("[PHASE] 3 completed!")
    # Phase 4
    forces = csd.get_action(mstar, gamma, say, pd.DataFrame.from_dict(sls_table))
    elfm_demands = csd.run_analysis(opt_sol, list(forces['Fi']))
    mphi_sections = csd.design_elements(elfm_demands, opt_sol, T_lower, T_upper)
    print("[PHASE] 4 completed!")
    # Phase 5
    period, phi = csd.run_ma(opt_sol, T_lower, T_upper, mphi_sections)
    print(period)
    csd.verify_period(round(period, 2), T_lower, T_upper)
    ductility_hard = csd.get_system_ductility(opt_sol, period, say[0], mphi_sections)
    print("[PHASE] 5 completed!")
