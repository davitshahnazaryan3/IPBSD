import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq


class Iterations:
    def __init__(self, ipbsd, sols, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath):
        """
        Initialize
        """
        self.ipbsd = ipbsd
        self.sols = sols
        self.spo_file = spo_file
        self.target_MAFC = target_MAFC
        self.analysis_type = analysis_type
        self.damping = damping
        self.num_modes = num_modes
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.spo_shape = None
        self.outputPath = outputPath
        self.omegaWarn = False
        self.warnT = True
        self.spo_validate = True
        self.model_outputs = None
        self.period_to_use = None
        self.modified = "spo"

    def compare_value(self, x, y, tol=0.05):
        """
        Verify whether x is within tolerance bounds of y
        :param x: float
        :param y: float
        :param tol: float
        :return: bool
        """
        if y - tol * y <= x <= y + tol * y:
            return True
        else:
            return False

    def compare_areas(self, x, y, tol=0.2):
        """

        :param x: dict
        :param y: dict
        :param tol: float
        :return: bool
        """
        peak_x = x["a"]*(x["mc"] - 1) + 1
        peak_y = y["a"]*(y["mc"] - 1) + 1

        a1x = (1 + peak_x)/2 * (x["mc"] - 1)
        a1y = (1 + peak_y)/2 * (y["mc"] - 1)

        a2x = (x["mf"] - x["mc"]) * peak_x / 2
        a2y = (y["mf"] - y["mc"]) * peak_y / 2

        return self.compare_value(a1x, a1y, tol=tol) and self.compare_value(a2x, a2y, tol=tol)

    def derive_spo_shape(self, spo, residual=0.1):
        """
        Fits a curve to the model SPO shape
        :param spo: dict                            Top displacement and base shear
        :param residual: float                      Percentage of Yield strength for residual
        :return: ndarrays                           Fitted top displacement and base shear
        """
        # Top displacement and base shear
        x = spo[0]
        y = spo[1]

        def getIndex(My, data):
            if np.where(data >= My)[0].size == 0:
                return np.nan
            else:
                return np.where(data >= My)[0][0]

        def getEquation(p1, p2):
            points = [p1, p2]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            return m, c

        # Get maximum moment point
        Vmax = max(y)
        dmax = x[getIndex(Vmax, y)]

        # Get initial stiffness
        m1 = 0.2 * Vmax
        d1 = x[getIndex(m1, y)]
        stiff_elastic = m1 / d1
        temp = y / x
        stfIdx = np.where(temp < 0.9 * stiff_elastic)[0][0]
        d2 = x[stfIdx]
        m2 = y[stfIdx]

        # Find point of plasticity initiation
        stf0 = (y[1] - y[0]) / (x[1] - x[0])
        for i in range(1, len(x)):
            stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if stf1 <= 0.5 * stf0:
                break
            else:
                stf0 = stf1

        if i == getIndex(Vmax, y):
            i = i - 5

        dPl = x[i]
        mPl = y[i]

        a0, b0 = getEquation((d1, m1), (d2, m2))
        a1, b1 = getEquation((dPl, mPl), (dmax, Vmax))

        # Find intersection point, i.e. the nominal yield point
        xint = (b1 - b0) / (a0 - a1)
        yint = a0 * xint + b0

        # Now, identify the residual strength point (here defined at V=0)
        yres = max(y[-1], yint * residual)
        idx = getIndex(1.01*yres, y[::-1])
        xres = x[::-1][idx]

        # Get the curve
        d = np.array([0., xint, dmax, xres])
        v = np.array([0., yint, Vmax, yres])

        return d, v

    def spo2ida_parameters(self, d, v, t):
        """
        Updates the SPO2IDA parameters
        :param d: ndarray                       Displacements
        :param v: ndarray                       Base shear forces
        :param t: float                         Fundamental period
        :return: dict                           Updated SPO2IDA input
        """
        # Residual strength
        r = v[-1] / v[1]
        # Hardening ductility
        muC = d[2] / d[1]
        # Fracturing ductility
        muF = d[3] / d[1]
        # Hardening slope
        a = (v[2] / v[1] - 1) / (muC - 1.)
        # Softening slope
        ap = (v[2] / v[1] - r) / (muC - muF)
        # Pinch weight
        pw = 1.0
        # Create a Dictionary
        spo_data = {"mc": muC, "a": a, "ac": ap, "r": r, "mf": muF, "pw": pw, "T": t}
        return spo_data

    def seek_solution(self, warnings, opt):
        """
        Seeks a solution within the already generated section combinations file if any warnings were recorded
        :param warnings: dict                       Dictionary of boolean warnings for each structural element
                                                    For any warning cross-section dimensions will be modified
        :param opt: Series                          Optimal solution
        :return: Series                             Solution containing c-s and modal properties
        :return: dict                               Modes corresponding to the solution for RSMA
        """
        # Number of storeys
        nst = self.ipbsd.data.nst

        # Remove column values
        cols_to_drop = ["T", "Weight", "Mstar", "Part Factor"]
        opt.loc[cols_to_drop] = np.nan

        # After modifications, equally between c-s of two storeys might not hold
        # Increment for cross-section modifications for elements with warnings
        increment = 0.05
        any_warnings = 0
        # Columns
        for ele in warnings["MAX"]["Columns"]:
            if warnings["MAX"]["Columns"][ele] == 1:
                inc = increment
                any_warnings = 1
            # elif warnings["MIN"]["Columns"][ele] == 1:
            #     inc = -increment
            #     any_warnings = 1
            else:
                inc = 0

            # Modify cross-section
            storey = ele[1]
            bay = int(ele[3])
            if bay == 1:
                opt[f"he{storey}"] = opt[f"he{storey}"] + inc

            else:
                opt[f"hi{storey}"] = opt[f"hi{storey}"] + inc

        # Beams
        for i in warnings["MAX"]["Beams"]:
            for ele in warnings["MAX"]["Beams"][i]:
                storey = ele[1]
                if warnings["MAX"]["Beams"][i][ele] == 1:
                    inc = increment
                    any_warnings = 1
                # elif warnings["MIN"]["Beams"][i][ele] == 1:
                #     inc = -increment
                #     any_warnings = 1
                else:
                    inc = 0

                # Increase section cross-section
                opt[f"b{storey}"] = opt[f"he{storey}"]
                opt[f"h{storey}"] = opt[f"h{storey}"] + inc

        # If any section was modified, we need to reapply the constraints
        if any_warnings == 1:
            '''Enforce constraints'''
            # Column storey constraints
            for st in range(1, nst, 2):
                if opt[f"he{st}"] != opt[f"he{st + 1}"]:
                    opt[f"he{st + 1}"] = opt[f"he{st}"]
                if opt[f"he{st}"] < opt[f"he{st + 1}"]:
                    opt[f"he{st}"] = opt[f"he{st + 1}"]

                if opt[f"hi{st}"] != opt[f"hi{st + 1}"]:
                    opt[f"hi{st + 1}"] = opt[f"hi{st}"]
                if opt[f"hi{st}"] < opt[f"hi{st + 1}"]:
                    opt[f"hi{st}"] = opt[f"hi{st + 1}"]

            # Column bay constraints
            for st in range(1, nst + 1):
                if opt[f"hi{st}"] > opt[f"he{st}"] + 0.2:
                    opt[f"he{st}"] = opt[f"hi{st}"] - 0.2

                if opt[f"hi{st}"] < opt[f"he{st}"]:
                    opt[f"hi{st}"] = opt[f"he{st}"]

            # Beam width and external column width constraint
            if opt[f"b{nst}"] != opt[f"he{nst}"]:
                opt[f"b{nst}"] = opt[f"he{nst}"] = max(opt[f"b{nst}"], opt[f"he{nst}"])

            # Beam equality constraint
            for st in range(1, nst, 2):
                if opt[f"b{st}"] != opt[f"b{st + 1}"]:
                    opt[f"b{st}"] = opt[f"b{st + 1}"] = max(opt[f"b{st}"], opt[f"b{st + 1}"])

                if opt[f"h{st}"] != opt[f"h{st + 1}"]:
                    opt[f"h{st}"] = opt[f"h{st + 1}"] = max(opt[f"h{st}"], opt[f"h{st + 1}"])

            # Max value limits
            for st in range(1, nst + 1):
                if opt[f"b{st}"] > 0.55:
                    opt[f"b{st}"] = 0.55
                if opt[f"h{st}"] > 0.75:
                    opt[f"h{st}"] = 0.75
                if opt[f"hi{st}"] > 0.75:
                    opt[f"hi{st}"] = 0.75
                if opt[f"he{st}"] > 0.75:
                    opt[f"he{st}"] = 0.75

            # Beam width to height ratio constraint
            for st in range(1, nst):
                if opt[f"b{st}"] > opt[f"h{st}"] - 0.1:
                    opt[f"h{st}"] = opt[f"b{st}"] + 0.1
                if opt[f"b{st}"] < opt[f"h{st}"] - 0.3:
                    opt[f"b{st}"] = opt[f"b{st}"] - 0.3

        # Finding a matching solution from the already generated DataFrame
        solution = self.sols[self.sols == opt].dropna(thresh=len(self.sols.columns) - 4)
        solution = self.sols.loc[solution.index]

        if solution.empty:
            raise ValueError("[EXCEPTION] No solution satisfying the period range condition was found!")
        else:
            solution, modes = self.ipbsd.get_all_section_combinations(t_lower=None, t_upper=None,
                                                                      solution=solution.iloc[0], data=self.ipbsd.data,
                                                                      cache_dir=self.outputPath / "Cache")
            return solution, modes

    def iterate_phase_3(self, opt_sol, omega, read=True):
        """
        Runs phase 3 of the framework
        :param opt_sol: Series                          Solution containing c-s and modal properties
        :param omega: float                             Overstrength ratio
        :param read: bool                               Whether to read the input file or not
        :return part_factor: float                      Participation factor of first mode
        :return m_star: float                           Effective first modal mass
        :return say: float                              Spectral acceleration at yield in g
        :return dy: float                               Spectral displacement at yield in m
        :return spo2ida_data: dict                      SPO2IDA results
        """
        """Perform SPO2IDA"""
        """Estimate parameters for SPO curve and compare with assumed shape"""
        # Set period equal to the actual period computed from MA or SPO analysis
        if self.period_to_use is not None:
            period = self.period_to_use
        else:
            period = round(float(opt_sol['T']), 1)

        if read:
            # Reads the input assumption if necessary
            self.spo_shape = self.ipbsd.data.initial_spo_data(period, self.spo_file)

        # Run SPO2IDA
        spo2ida_data = self.ipbsd.perform_spo2ida(self.spo_shape)
        print("[SUCCESS] SPO2IDA was performed")

        """Yield strength optimization for MAFC and verification"""
        part_factor = opt_sol["Part Factor"]
        m_star = opt_sol["Mstar"]
        cy, dy = self.ipbsd.verify_mafc(period, spo2ida_data, part_factor, self.target_MAFC, omega, hazard="True")

        print("[SUCCESS] MAFC was validated")
        return part_factor, m_star, cy, dy, spo2ida_data

    def iterate_phase_4(self, cy, dy, sa, period_range, solution, modes, table_sls, rerun=False):
        """
        Runs phase 4 of the framework
        :param cy: float                                Spectral acceleration at yield in g
        :param dy: float                                Spectral displacement at yield in m
        :param sa: list                                 Spectral accelerations in g of the spectrum
        :param period_range: list                       Periods of the spectrum
        :param solution: Series                         Solution containing c-s and modal properties
        :param modes: dict                              Periods and normalized modal shapes of the solution
        :param table_sls: DataFrame                     Table with SLS parameters
        :param rerun: bool                              Perform reruns for stiffness correction or not
        :return forces: DataFrame                       Acting forces
        :return demands: dict                           Demands on the structural elements
        :return details: dict                           Designed element properties from the moment-curvature
                                                        relationship
        :return hard_ductility: float                   Estimated system hardening ductility
        :return fract_ductility: float                  Estimated system fracturing ductility
        :return warn: bool                              Indicates whether any warnings were displayed
        :return warnings: dict                          Dictionary of boolean warnings for each structural element
                                                        i.e. local ductility requirements not met in Detailing
        """
        # Get Sa at each modal period of interest
        if self.analysis_type == 4 or self.analysis_type == 5:
            se_rmsa = self.ipbsd.get_sa_at_period(cy, sa, period_range, modes["Periods"])
        else:
            se_rmsa = None

        self.num_modes = min(self.num_modes, self.ipbsd.data.nst)
        if self.analysis_type == 4 or self.analysis_type == 5:
            corr = self.ipbsd.get_correlation_matrix(modes["Periods"], self.num_modes, damping=self.damping)
        else:
            corr = None
        forces = self.ipbsd.get_action(solution, cy, pd.DataFrame.from_dict(table_sls), self.ipbsd.data.w_seismic,
                                       self.analysis_type, self.num_modes, modes, modal_sa=se_rmsa)

        print("[SUCCESS] Actions on the structure for analysis were estimated")
        # if self.export_cache:
        #     self.export_results(self.outputPath / "Cache/action", forces, "csv")
        yield forces

        """Perform ELF analysis"""
        def analyze(hinge=None):
            if self.analysis_type == 1:
                demands = self.ipbsd.run_muto_approach(solution, list(forces["Fi"]), self.ipbsd.data.h,
                                                       self.ipbsd.data.spans_x)
            elif self.analysis_type == 2:
                demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), fstiff=self.fstiff,
                                                  hinge=hinge)
            elif self.analysis_type == 3:
                demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                                  fstiff=self.fstiff, hinge=hinge)
            elif self.analysis_type == 4 or self.analysis_type == 5:
                demands = {}
                for mode in range(self.num_modes):
                    demands[f"Mode{mode + 1}"] = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                                         list(forces["Fi"][:, mode]),
                                                                         fstiff=self.fstiff, hinge=hinge)

                demands = self.ipbsd.perform_cqc(corr, demands)

                if self.analysis_type == 5:
                    demands_gravity = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                              grav_loads=list(forces["G"]),
                                                              fstiff=self.fstiff, hinge=hinge)
                    # Combining gravity and RSMA results
                    for eleType in demands_gravity.keys():
                        for dem in demands_gravity[eleType].keys():
                            if eleType == "Beams" and dem == "M":
                                demands[eleType][dem]["Pos"] = demands[eleType][dem]["Pos"] + \
                                                               demands_gravity[eleType][dem]["Pos"]
                                demands[eleType][dem]["Neg"] = demands[eleType][dem]["Neg"] + \
                                                               demands_gravity[eleType][dem]["Neg"]
                            else:
                                demands[eleType][dem] = demands[eleType][dem] + demands_gravity[eleType][dem]

            else:
                raise ValueError("[EXCEPTION] Incorrect analysis type...")
            return demands

        demands = analyze(hinge=None)
        print("[SUCCESS] Analysis completed and demands on structural elements were estimated.")

        """Design the structural elements"""
        details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
            self.ipbsd.design_elements(demands, solution, modes, dy, cover=self.rebar_cover)

        print("[SUCCESS] Section detailing done. Element idealized Moment-Curvature relationships obtained")
        # Rerun elastic analysis with updated stiffness reduction factors for all structural elements
        if rerun:
            demands = analyze(hinge=hinge_models)

            details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
                self.ipbsd.design_elements(demands, solution, modes, dy, cover=self.rebar_cover)

            print("[RERUN COMPLETE] Rerun of analysis and detailing complete due to modified stiffness.")

        yield demands
        yield details, hinge_models, mu_c, mu_f
        yield warnMax, warnMin, warnings

    def run_ma(self, opt_sol, hinge, forces, t_upper, tol=0.05):
        """
        Creates a nonlinear model and runs Modal Analysis with the aim of correcting opt_sol and fundamental period
        :param opt_sol: DataFrame                   Optimal solution
        :param hinge: DataFrame                     Hinge models after detailing
        :param forces: DataFrame                    Acting forces (gravity, lateral)
        :param t_upper: float                       Upper period limit from IPBSD
        :param tol: float                           Tolerance of upper period limit satisfaction
        :return model_periods: ndarray              Model periods from MA
        :return modalShape: list                    Modal shapes from MA
        :return gamma: float                        First mode participation factor from MA
        :return mstar: float                        First mode effective mass from MA
        :return opt_sol: DataFrame                  Corrected optimal solution (new c-s and T1)
        """
        model_periods, modalShape, gamma, mstar = self.ipbsd.ma_analysis(opt_sol, hinge, forces, self.fstiff)

        # There is a high likelihood that Fundamental period and SPO curve shape will not match the assumptions
        # Therefore the first iteration should correct both assumptions (further corrections are likely not to be large)
        if model_periods[0] > t_upper + model_periods[0] * tol:
            # Period error
            period_error = model_periods[0] - opt_sol["T"]
            # Look for a new period for the design solution (probably not the best way to guess)
            tnew = t_upper - period_error
            # Select all solutions in the vicinity of the new period
            sols = self.sols[(self.sols["T"] >= tnew - tnew * tol) & (self.sols["T"] <= tnew + tnew * tol)]
            # Select the solution with the least weight
            opt_sol = sols[sols["Weight"] == sols["Weight"].min()].iloc[0]
            # Actual period of the structure is guessed to be (opt_sol period + error)
            self.period_to_use = opt_sol["T"] + period_error
            self.warnT = True

        else:
            # If the model period was within the confidence
            self.period_to_use = model_periods[0]
            self.warnT = False

        return model_periods, modalShape, gamma, mstar, opt_sol

    def run_spo(self, opt_sol, hinge_models, forces, vy, modalShape, omega):
        """
        Create a nonlinear model in OpenSees and runs SPO
        :param opt_sol: DataFrame               Design solution
        :param hinge_models: DataFrame          Nonlinear hinge models
        :param forces: DataFrame                Acting loads
        :param vy: float                        Design base shear of the MDOF system (excludes Omega)
        :param modalShape: list                 First-mode shape from MA
        :param omega: float                     Overstrength factor
        :return: tuple                          SPO outputs (Top displacement vs. Base Shear)
        :return: tuple                          Idealized SPO curve fit (Top displacement vs. Base Shear)
        :return: float                          Overstrength factor
        """
        spoResults = self.ipbsd.spo_opensees(opt_sol, hinge_models, forces, self.fstiff, modalShape)

        # Get the idealized version of the SPO curve and create a warningSPO = True if the assumed shape was
        # incorrect
        d, v = self.derive_spo_shape(spoResults, residual=0.1)

        # Actual overstrength
        omegaNew = v[1] / vy

        # Check if the new overstrength is correct
        if omega * 0.95 <= omegaNew <= omega * 1.05:
            omegaNew = omega
            self.omegaWarn = False
        else:
            # if self.modified == "spo":
            #     omegaNew = omegaNew
            # else:
            #     omegaNew = omega
            self.omegaWarn = True

        # There is a high likelihood that Fundamental period and SPO curve shape will not match the assumptions.
        # Therefore the first iteration should correct both assumptions (further corrections are likely not to be large)
        # Get new SPO parameters
        spo_data_updated = self.spo2ida_parameters(d, v, self.period_to_use)

        # Check whether the parameters vary from the original assumption (if True, then tolerance met)
        # NOTE: spo shape might loop around two scenarios (need a better technique to look for a solution)
        # It is primarily due to some low values (e.g. muC), where the large variations of e.g. a do not have a
        # significant impact on the final results, however are hard to fit
        # self.spo_validate = all(list(map(self.compare_value, self.spo_data.values(), spo_data_updated.values())))

        # Alternatively compare the area under the normalized SPO curves (possible update: compare actual SPO curves)
        self.spo_validate = self.compare_areas(self.spo_shape, spo_data_updated, tol=0.1)

        # Update the SPO parameters
        if not self.spo_validate:
            # if self.modified == "overstrength":
            #     self.spo_shape = spo_data_updated
            # else:
            #     self.spo_shape = self.spo_shape
            self.spo_shape = spo_data_updated
        else:
            self.spo_shape = self.spo_shape

        # Change value of 'modified'
        if self.modified == "spo":
            self.modified = "overstrength"
        else:
            self.modified = "spo"

        return spoResults, [d, v], omegaNew

    def validations(self, opt_sol, modes, sa, period_range, table_sls, t_lower, t_upper, iterate=True, maxiter=20,
                    omega=None):
        """
        Runs the iterations
        :param opt_sol: DataFrame                       Optimal solution (design solution)
        :param modes: dict                              Modes for RMSA from optimal solution
        :param sa: ndarray                              Spectral accelerations at SLS
        :param period_range: ndarray                    Periods at SLS
        :param table_sls: dict                          Outputs of DBD at SLS
        :param t_lower: float                           Lower period
        :param t_upper: float                           Upper period
        :param iterate: bool                            Whether iterations to seek for better solutions are necessary
        :param maxiter: int                             Maximum number of iterations before the framework halts
        :param omega: float                             Overstrength ratio
        :return: ipbsd_outputs: dict                    IPBSD outputs to cache
        :return: spo2ida_data: dict                     SPO2IDA outputs to cache
        :return: opt_sol: df.Series                     Optimal solution to cache
        :return: demands: dict                          Demands on structural elements to cache
        :return: details: dict                          Detailing outputs to cache
        :return: hinge_models: DataFrame                Hysteretic hinge model parameters
        :return: forces: DataFrame                      Acting loads and masses on the structure
        """
        '''
        MAIN POINTS LEADING THE OPTIMIZATION PROCESS:
        * Increase the reinforcement cover, thus requiring more reinforcement, and decreasing the curvature ductility
        * Keep reinforcement ratio constant, while increasing the cross-section dimension, thus increasing capacity and
        curvature ductility, while reducing fundamental period
        * Take a route of avoiding overstrength due to forced minimum reinforcement ratio requirement
        * Less is better, i.e. fewer iterations to avoid non-convergence issues, where overstrength and ductility 
        increase hand in hand
        * If ductility is high, then cy required for MAFC will be small, small cy, might result in higher overstrength,
        since demands will be lower, while the sections will remain the same
        * Reducing c-s might result in higher T1, which we want to avoid, since it is limited by Tupper

        VARIABLES: 
        Cy = cy, design yield strength of ESDOF
        Omega = omega, overstrength factor
        SPO shape (ductilities, hardening, softening) = self.spo_data, parameters representing SPO curve shape
        T1 = self.period_to_use and opt_sol, fundamental period
        c-s = opt_sol, cross section dimensions of structural elements

        WARNINGS: warnMin - min reinforcement ratio limit reached; warnMax - max reinforcement ratio limit reached;
        warnT - fundamental period not met, c-s being modified; 
        THE PROCESS:
        0. Selected c-s and T1 (assumed, since fstiff of 0.5 might not stand for all elements)
        1. Assume SPO shape, find Cy to meet MAFC
        2. With Cy and c-s found, find demands 
        3. Do detailing (cover, reinforcement layers will affect the Ductility)
        ---- Fist correction: return and rerun analysis with updated fstiff of all elements, rerun from 2 to 3.
        4. Create nonlinear model and run modal analysis (T1 might change here, hence c-s will change as well)
        ---- Second correction: fix c-s, T1 and rerun from 0 to 4.  warnT
        ---- Third correction: if unsatisfactory detailing, change c-s dimensions, rerun 2 to 4. warnMax or warnMin
        5. Create nonlinear model and run static pushover analysis (SPO shape and overstrength might change here)
        ---- Fourth correction: if SPO shape or overstrength change, rerun from 1 to 5
        '''

        # Initialize period to use
        self.period_to_use = None

        print("[PHASE] Commencing phase 3...")
        gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omega)

        """Get action and demands"""
        print("[PHASE] Commencing phase 4...")
        phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls)
        forces = next(phase_4)
        demands = next(phase_4)
        details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
        warnMax, warnMin, warnings = next(phase_4)
        # Reading SPO parameters from file
        read = True

        if iterate:
            """Initialize iterations for corrections of assumptions and optimizations"""
            """
            :param iterate: bool                    If iterations are needed (if True, do iterations)
            :param warn: bool                       Detailing warnings of structural elements (if True, run iterations)
            :param spo_validate: bool               SPO curve shape and fundamental period (if False, run iterations)
            :param omegaWarn: bool                  Overstrength assumption check (if True, run iterations)
            """
            # Start with the correction of SPO curve shape. Also, recalculate the mode shape of new design solution
            cnt = 0
            periodCache = opt_sol["T"]

            # Correction for period
            """Create an OpenSees model and run modal analysis"""
            model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces, t_upper)
            # Update modes
            modes["Periods"] = np.array(model_periods)
            modes["Modes"][0, :] = np.array(modalShape)

            # Iterate until all conditions are met
            while (self.warnT or warnMax or not self.spo_validate or self.omegaWarn) and cnt + 1 <= maxiter:

                # Iterations related to SPO corrections
                if not self.spo_validate or self.omegaWarn:
                    # Reruns
                    print("[RERUN] Rerun for SPO shape correction...")
                    # Calculates the new cy for the corrected SPO shape, period, Overstrength and c-s
                    gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omega, read=read)
                    # Run elastic analysis and detail the structure
                    phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls)
                    forces = next(phase_4)
                    demands = next(phase_4)
                    details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
                    warnMax, warnMin, warnings = next(phase_4)

                    # Run modal analysis to check the T1
                    model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                   t_upper)
                    # Update modes
                    modes["Periods"] = np.array(model_periods)
                    modes["Modes"][0, :] = np.array(modalShape)
                    print("[RERUN COMPLETE] Rerun for SPO shape correction.")

                # Iterations related to Period corrections
                cntTrerun = 0
                while self.warnT:
                    if cntTrerun > 0:
                        rerun = True
                    else:
                        rerun = False

                    # Reruns
                    print("[RERUN] Rerun for fundamental period correction...")
                    gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omega, read=read)
                    phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls, rerun=rerun)
                    forces = next(phase_4)
                    demands = next(phase_4)
                    # TODO, increasing cover results in issues, fix (maybe fixed with recent changes, verify)
                    details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
                    warnMax, warnMin, warnings = next(phase_4)

                    # Run modal analysis
                    model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                   t_upper)
                    # Update modes
                    modes["Periods"] = np.array(model_periods)
                    modes["Modes"][0, :] = np.array(modalShape)

                    periodCache = opt_sol["T"]
                    cntTrerun += 1
                    if not self.warnT:
                        # Update optimal solution (the update difference should be minimal as significant changes
                        # not expected)
                        opt_sol["Mstar"] = mstar
                        opt_sol["Part Factor"] = gamma
                        opt_sol["T"] = model_periods[0]
                        # TODO, rerun for new cy, detailing and MA. Or significant changes not expected?
                        print("[RERUN COMPLETE] Rerun for fundamental period correction")

                # Exiting while warnT
                # Correction if unsatisfactory detailing, modifying only towards increasing c-s
                if warnMax:
                    """Look for a different solution"""
                    # Get the new design solution and the modal shapes
                    opt_sol, modes = self.seek_solution(warnings, opt_sol)
                    print("[RERUN COMPLETE] New design solution has been selected due to unsatisfactory detailing...")

                """Create an OpenSees model and run static pushover analysis"""
                vy_assumed = cy * gamma * mstar * 9.81
                omegaCache = omega
                spoShapeCache = self.spo_shape

                spoResults, spo_idealized, omega = self.run_spo(opt_sol, hinge_models, forces, vy_assumed, modalShape,
                                                                omega)

                # Record OpenSees outputs
                self.model_outputs = {"MA": {"T": model_periods, "modes": modalShape, "gamma": gamma, "mstar": mstar},
                                      "SPO": spoResults, "SPO_idealized": spo_idealized}

                # Reading SPO parameters from file
                read = False

                print("[SUCCESS] Static pushover analysis was successfully performed.")

                # Print out information
                print("--------------------------")
                print(f"[ITERATION {cnt + 1} END] Actual over assumed values of variables are provided: \n"
                      f"Yield strength overstrength: {omega / omegaCache * 100:.0f}%, \n"
                      f"Hardening ductility: {self.spo_shape['mc'] / spoShapeCache['mc'] * 100:.0f}%, \n"
                      f"Fracturing ductility: {self.spo_shape['mf'] / spoShapeCache['mf'] * 100:.0f}%, \n"
                      f"Fundamental period: {opt_sol['T'] / periodCache:.0f}%.")
                print("--------------------------")

                # Increase count of iterations
                cnt += 1

            if cnt == maxiter:
                print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": gamma, "Mstar": mstar, "Period range": [t_lower, t_upper],
                         "overstrength": omega, "yield": [cy, dy], "muc": float(self.spo_shape["mc"])}

        return ipbsd_outputs, spo2ida_data, opt_sol, demands, details, hinge_models, forces, self.model_outputs
