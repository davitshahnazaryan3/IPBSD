import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq


class Iterations:
    def __init__(self, ipbsd, sols, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath, gravity_loads, flag3d=False):
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
        self.spo_validate = False
        self.model_outputs = None
        self.period_to_use = None
        self.modified = "spo"
        self.gravity_loads = gravity_loads
        self.flag3d = flag3d

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

    def seek_solution(self, warnings, opt_sol, direction="x"):
        """
        Seeks a solution within the already generated section combinations file if any warnings were recorded
        :param warnings: dict                       Dictionary of boolean warnings for each structural element
                                                    For any warning cross-section dimensions will be modified
        :param opt_sol: Series                      Optimal solution
        :param direction: str                       Direction to look for the solution
        :return: Series                             Solution containing c-s and modal properties
        :return: dict                               Modes corresponding to the solution for RSMA
        """
        # Get the solution of interest
        if self.flag3d:
            opt = opt_sol[direction + "_seismic"]
        else:
            opt = opt_sol

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

        # Solutions to look for
        if self.flag3d:
            if direction == "x":
                solution_x = solution.iloc[0]
                solution_y = opt_sol["y_seismic"]
            else:
                solution_x = opt_sol["x_seismic"]
                solution_y = solution.iloc[0]
        else:
            solution_x = solution.iloc[0]
            solution_y = None

        if solution.empty:
            raise ValueError("[EXCEPTION] No solution satisfying the period range condition was found!")
        else:
            results = self.ipbsd.get_all_section_combinations(period_limits=None, solution_x=solution_x,
                                                              solution_y=solution_y, data=self.ipbsd.data,
                                                              cache_dir=self.outputPath / "Cache")

            if self.flag3d:
                solution = results[0]["opt_sol"] if direction == "x" else results[1]["opt_sol"]
                modes = results[0]["opt_modes"] if direction == "x" else results[1]["opt_modes"]
            else:
                solution = results["opt_sol"]
                modes = results["opt_modes"]

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

    def iterate_phase_4(self, cy, dy, sa, period_range, solution, modes, table_sls, rerun=False, direction=0):
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
        :param direction: int                           0 for X, 1 for Y
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
        # Initialize hinge models
        hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

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

        # Gravity loads
        if self.flag3d:
            seismic_solution = solution["x_seismic"] if direction == 0 else solution["y_seismic"]
        else:
            seismic_solution = solution

        forces = self.ipbsd.get_action(seismic_solution, cy, pd.DataFrame.from_dict(table_sls), self.gravity_loads,
                                       self.analysis_type, self.num_modes, modes, modal_sa=se_rmsa)

        print("[SUCCESS] Actions on the structure for analysis were estimated")

        # if self.export_cache:
        #     self.export_results(self.outputPath / "Cache/action", forces, "csv")
        yield forces

        """Perform ELF analysis"""
        def analyze(hinge=None):
            if hinge is None and self.flag3d:
                hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

            if self.analysis_type == 1:
                if direction == 0:
                    spans = self.ipbsd.data.spans_x
                else:
                    spans = self.ipbsd.data.spans_y

                demands = self.ipbsd.run_muto_approach(solution, list(forces["Fi"]), self.ipbsd.data.h, spans)
            elif self.analysis_type == 2:
                demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), fstiff=self.fstiff,
                                                  hinge=hinge, direction=direction)
            elif self.analysis_type == 3:
                demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                                  fstiff=self.fstiff, hinge=hinge, direction=direction)

            elif self.analysis_type == 4 or self.analysis_type == 5:
                demands = {}
                for mode in range(self.num_modes):
                    demands[f"Mode{mode + 1}"] = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                                         list(forces["Fi"][:, mode]),
                                                                         fstiff=self.fstiff, hinge=hinge,
                                                                         direction=direction)

                demands = self.ipbsd.perform_cqc(corr, demands)

                if self.analysis_type == 5:
                    demands_gravity = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                              grav_loads=list(forces["G"]),
                                                              fstiff=self.fstiff, hinge=hinge, direction=direction)

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
        # Details the frame of seismic direction only
        seismic_demands = demands["x_seismic"] if direction == 0 else demands["y_seismic"]
        details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
            self.ipbsd.design_elements(seismic_demands, seismic_solution, modes, dy, cover=self.rebar_cover,
                                       direction=direction)
        print("[SUCCESS] Section detailing done. Element idealized Moment-Curvature relationships obtained")

        # Rerun elastic analysis with updated stiffness reduction factors for all structural elements
        if rerun:
            if self.flag3d:
                # Store the results into hinge and rerun analysis with updated stiffness
                if direction == 0:
                    hinge["x_seismic"] = hinge_models
                else:
                    hinge["y_seismic"] = hinge_models
                demands = analyze(hinge=hinge)
            else:
                demands = analyze(hinge=hinge_models)

            details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
                self.ipbsd.design_elements(seismic_demands, solution, modes, dy, cover=self.rebar_cover,
                                           direction=direction)

            print("[RERUN COMPLETE] Rerun of analysis and detailing complete due to modified stiffness.")

        if self.flag3d:
            if direction == 0:
                hinge["x_seismic"] = hinge_models
            else:
                hinge["y_seismic"] = hinge_models
            demands_gravity = demands["gravity"]
        else:
            demands_gravity = None

        yield demands, demands_gravity
        yield details, hinge_models, mu_c, mu_f
        yield warnMax, warnMin, warnings

    def run_ma(self, opt_sol, hinge, forces, t_upper, tol=0.05, direction="x"):
        """
        Creates a nonlinear model and runs Modal Analysis with the aim of correcting opt_sol and fundamental period
        :param opt_sol: DataFrame                   Optimal solution
        :param hinge: DataFrame                     Hinge models after detailing
        :param forces: DataFrame                    Acting forces (gravity, lateral)
        :param t_upper: float                       Upper period limit from IPBSD
        :param tol: float                           Tolerance of upper period limit satisfaction
        :param direction: str                       Direction of action (x for 2D, x or y for 3D)
        :return model_periods: ndarray              Model periods from MA
        :return modalShape: list                    Modal shapes from MA
        :return gamma: float                        First mode participation factor from MA
        :return mstar: float                        First mode effective mass from MA
        :return opt_sol: DataFrame                  Corrected optimal solution (new c-s and T1)
        """
        print("[MA] Running modal analysis")
        model_periods, modalShape, gamma, mstar = self.ipbsd.ma_analysis(opt_sol, hinge, forces, self.fstiff)

        if self.flag3d:
            if direction == "x":
                if opt_sol["x_seismic"]["T"] >= opt_sol["y_seismic"]["T"]:
                    idx = 0
                else:
                    idx = 1
                seismic_solution = opt_sol["x_seismic"]
            else:
                if opt_sol["x_seismic"]["T"] >= opt_sol["y_seismic"]["T"]:
                    idx = 1
                else:
                    idx = 0
                seismic_solution = opt_sol["y_seismic"]
            model_periods = model_periods[idx]
            modalShape = modalShape[:, idx]
            gamma = gamma[idx]
            mstar = mstar[idx]
        else:
            # Interested only in first mode for the 2D approach
            model_periods = model_periods[0]
            seismic_solution = opt_sol

        # There is a high likelihood that Fundamental period and SPO curve shape will not match the assumptions
        # Therefore the first iteration should correct both assumptions (further corrections are likely not to be large)
        if model_periods > t_upper + model_periods * tol:
            # Period error
            period_error = model_periods - seismic_solution["T"]
            # Look for a new period for the design solution (probably not the best way to guess)
            tnew = t_upper - period_error
            # Select all solutions in the vicinity of the new period
            sols = self.sols[(self.sols["T"] >= tnew - tnew * tol) & (self.sols["T"] <= tnew + tnew * tol)]
            # Select the solution with the least weight
            seismic_solution = sols[sols["Weight"] == sols["Weight"].min()].iloc[0]
            # Actual period of the structure is guessed to be (opt_sol period + error)
            self.period_to_use = seismic_solution["T"] + period_error
            self.warnT = True

            if self.flag3d:
                # Update the optimal solution
                if direction == "x":
                    opt_sol["x_seismic"] = seismic_solution
                else:
                    opt_sol["y_seismic"] = seismic_solution
            else:
                opt_sol = seismic_solution

        else:
            # If the model period was within the confidence
            self.period_to_use = model_periods
            self.warnT = False

        return model_periods, modalShape, gamma, mstar, opt_sol

    def run_spo(self, opt_sol, hinge_models, forces, vy, modalShape, omega, direction="x"):
        """
        Create a nonlinear model in OpenSees and runs SPO
        :param opt_sol: DataFrame               Design solution
        :param hinge_models: DataFrame          Nonlinear hinge models
        :param forces: DataFrame                Acting loads
        :param vy: float                        Design base shear of the MDOF system (excludes Omega)
        :param modalShape: list                 First-mode shape from MA
        :param omega: float                     Overstrength factor
        :pararm direction: str of               Direction of pushover action
        :return: tuple                          SPO outputs (Top displacement vs. Base Shear)
        :return: tuple                          Idealized SPO curve fit (Top displacement vs. Base Shear)
        :return: float                          Overstrength factor
        """
        print("[SPO] Starting SPO analysis")
        d = 0 if direction == "x" else 1
        spoResults = self.ipbsd.spo_opensees(opt_sol, hinge_models, forces, self.fstiff, modalShape, direction=d)

        # Get the idealized version of the SPO curve and create a warningSPO = True if the assumed shape was incorrect
        d, v = self.derive_spo_shape(spoResults, residual=0.1)

        # Actual overstrength
        if self.flag3d:
            # Assumption: since only perimeter seismic frames are supported, we will always have 2 seismic frames
            factor = 0.5
        else:
            factor = 1.0
        omegaNew = v[1] / vy * factor

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

    def generate_initial_solutions(self, opt_sol, opt_modes, omega, sa, period_range, table_sls):
        """
        Master file to run for each direction/iteration
        :return:
        """
        # Call the iterations function (iterations have not yet started though)
        # 3D option
        frames = ["x", "y"]
        # Initialize dictionary for storing design outputs
        design_outputs = {"x": {}, "y": {}, "gravity": {}}
        demands_gravity = {}

        for i in frames:

            solution = opt_sol[f"{i}_seismic"]
            modes = opt_modes[i]
            table = table_sls[i]
            direction = 0 if i == "x" else 1

            if i == "x":
                # Applying lateral loads in X direction
                print("[INITIATE] Designing frame in X direction!")
            else:
                # Applying lateral loads in Y direction
                print("[INITIATE] Designing frame in Y direction!")

            # Initialize period to use
            self.period_to_use = None

            # Generate initial solutions for both directions
            # For each primary direction of building, apply lateral loads and design the structural elements
            print("[PHASE] Commencing phase 3...")
            gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(solution, omega)

            """Get action and demands"""
            print("[PHASE] Commencing phase 4...")
            phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table, direction=direction)
            forces = next(phase_4)
            demands, demands_gravity[i] = next(phase_4)
            details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
            warnMax, warnMin, warnings = next(phase_4)

            # Append information to design solutions
            design_outputs[i]["forces"] = forces
            design_outputs[i]["demands"] = demands
            design_outputs[i]["details"] = details
            design_outputs[i]["hinge_models"] = hinge_models
            design_outputs[i]["warnings"] = {"warnMax": warnMax, "warnMin": warnMin, "warnings": warnings}
            design_outputs[i]["phase3"] = {"cy": cy, "dy": dy, "spo2ida_data": spo2ida_data, "gamma": gamma,
                                           "mstar": mstar}

        # Detail the gravity frames in either direction (envelope)
        hinge_gravity = self.ipbsd.design_elements(demands_gravity, opt_sol["gravity"], None, None,
                                                   cover=self.rebar_cover, direction=0, gravity=True)
        design_outputs["gravity"]["hinge_models"] = hinge_gravity

        return design_outputs

    def run_iterations_for_3d(self, design_outputs, period_limits, opt_sol, opt_modes, sa, period_range, table_sls,
                              iterate=True, maxiter=20, omega=None):
        """
        Runs iterations for 3D approach in both directions of action
        :param design_outputs: dict
        :param period_limits: dict
        :param opt_sol: DataFrame
        :param opt_modes: dict
        :param sa: ndarray
        :param period_range: ndarray
        :param table_sls: dict
        :param iterate: bool
        :param maxiter: int
        :param omega: float
        :return: dict
        """
        outputs = {"ipbsd_outputs": {}, "spoResults": {}, "opt_sol": {}, "demands": {}, "details": {},
                   "hinge_models": {}, "action": {}, "modelOutputs": {}}
        for i in ["x", "y"]:
            # TODO, redesign external seismic columns only if the new demands are larger than the old ones
            print(f"[INITIATE 3D] Running framework in {i} direction!")
            table = table_sls[i]
            periods = period_limits[i]
            ipbsd_outputs, spo2ida_data, opt_sol, demands, details, hinge_models, forces, model_outputs = \
                self.validations(opt_sol, opt_modes, sa, period_range, table, periods, iterate=iterate, maxiter=maxiter,
                                 omega=omega, direction=i, initial_design_sols=design_outputs)

            outputs["ipbsd_outputs"][i] = ipbsd_outputs
            outputs["spoResults"][i] = spo2ida_data
            outputs["opt_sol"][i] = opt_sol
            outputs["demands"][i] = demands
            outputs["details"][i] = details
            outputs["hinge_models"][i] = hinge_models
            outputs["action"][i] = forces
            outputs["modelOutputs"][i] = model_outputs

            # Reset global variables
            self.spo_shape = None
            self.omegaWarn = False
            self.warnT = True
            self.spo_validate = False
            self.model_outputs = None
            self.period_to_use = None

        return outputs

    def validations(self, opt_sol, modes, sa, period_range, table_sls, period_limits, iterate=True, maxiter=20,
                    omega=None, direction="x", initial_design_sols=None):
        """
        Runs the iterations
        :param opt_sol: DataFrame                       Optimal solution (design solution)
        :param modes: dict                              Modes for RMSA from optimal solution
        :param sa: ndarray                              Spectral accelerations at SLS
        :param period_range: ndarray                    Periods at SLS
        :param table_sls: dict                          Outputs of DBD at SLS
        :param period_limits: list                      Lower and upper period bounds
        :param iterate: bool                            Whether iterations to seek for better solutions are necessary
        :param maxiter: int                             Maximum number of iterations before the framework halts
        :param omega: float                             Overstrength ratio
        :param direction: str                           Direction of action
        :param initial_design_sols: dict                Initial design solutions (for 3D approach only)
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
        # Upper period bound
        t_upper = period_limits[1]

        if not self.flag3d:
            # 2D frame consideration only
            # Initialize period to use
            self.period_to_use = None

            # Generate initial solutions for both directions
            # For each primary direction of building, apply lateral loads and design the structural elements
            print("[PHASE] Commencing phase 3...")
            gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omega)

            """Get action and demands"""
            print("[PHASE] Commencing phase 4...")
            phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls)
            forces = next(phase_4)
            demands, demands_gravity = next(phase_4)
            details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
            warnMax, warnMin, warnings = next(phase_4)

        else:
            # Initial design solutions for 3D approach
            forces = initial_design_sols[direction]["forces"]
            hinge_models = {"x_seismic": initial_design_sols["x"]["hinge_models"],
                            "y_seismic": initial_design_sols["y"]["hinge_models"],
                            "gravity": initial_design_sols["gravity"]["hinge_models"]}
            warnMax = initial_design_sols[direction]["warnings"]["warnMax"]
            warnMin = initial_design_sols[direction]["warnings"]["warnMin"]
            warnings = initial_design_sols[direction]["warnings"]["warnings"]
            cy = initial_design_sols[direction]["phase3"]["cy"]
            dy = initial_design_sols[direction]["phase3"]["dy"]
            spo2ida_data = initial_design_sols[direction]["phase3"]["spo2ida_data"]
            gamma = initial_design_sols[direction]["phase3"]["gamma"]
            mstar = initial_design_sols[direction]["phase3"]["mstar"]
            demands = initial_design_sols[direction]["demands"]
            details = initial_design_sols[direction]["details"]

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
            # Star the counter
            cnt = 0

            # Correction for period
            """Create an OpenSees model and run modal analysis"""
            model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces, t_upper,
                                                                           direction=direction)

            # Reread initial assumption of pushover shape (Same as for X direction)
            if direction == "y":
                self.spo_shape = self.ipbsd.data.initial_spo_data(self.period_to_use, self.spo_file)

            # Update modes and some utility variables
            if self.flag3d:
                idx = 0 if direction == "x" else 1
                # Modal properties
                modes[direction]["Periods"] = np.array([model_periods])
                modes[direction]["Modes"][idx, :] = np.array(modalShape)
                # Spo load pattern as first-mode proportional
                spoPattern = np.abs(modes[direction]["Modes"][idx, :])
                # Seismic solution
                solution = opt_sol[direction + "_seismic"]
                # Seismic direction tag
                tag = direction + "_seismic"
                # Period to cache
                periodCache = opt_sol[tag]["T"]
            else:
                idx = 0
                modes["Periods"] = np.array(model_periods)
                modes["Modes"][0, :] = np.array(modalShape)
                spoPattern = np.abs(modalShape)
                solution = opt_sol
                tag = None
                periodCache = opt_sol["T"]

            # Iterate until all conditions are met
            while (self.warnT or warnMax or not self.spo_validate or self.omegaWarn) and cnt + 1 <= maxiter:

                # Iterations related to SPO corrections (skip first iteration, before running SPO)
                if (not self.spo_validate or self.omegaWarn) and cnt > 0:
                    # Not being run at first iteration
                    # Reruns
                    print("[RERUN] Rerun for SPO shape correction...")
                    # Calculates the new cy for the corrected SPO shape, period, Overstrength and c-s
                    gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(solution, omega, read=read)

                    # Run elastic analysis and detail the structure
                    modes_to_use = modes[direction] if self.flag3d else modes
                    phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes_to_use, table_sls,
                                                   direction=idx)
                    forces = next(phase_4)
                    demands, temp = next(phase_4)
                    details, new_hinge_model, hard_ductility, fract_ductility = next(phase_4)
                    warnMax, warnMin, warnings = next(phase_4)
                    if self.flag3d:
                        # Update the hinge models related to the direction of interest
                        hinge_models[tag] = new_hinge_model
                    else:
                        hinge_models = new_hinge_model

                    # Run modal analysis to check the T1
                    model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                   t_upper, direction=direction)
                    # Update modes
                    if self.flag3d:
                        modes[direction]["Periods"] = np.array([model_periods])
                        modes[direction]["Modes"][idx, :] = np.array(modalShape)
                        spoPattern = np.abs(modes[direction]["Modes"][idx, :])
                    else:
                        modes["Periods"] = np.array(model_periods)
                        modes["Modes"][0, :] = np.array(modalShape)
                        spoPattern = np.abs(modalShape)

                    print("[RERUN COMPLETE] Rerun for SPO shape correction.")

                # Iterations related to Period corrections, only if run MA indicated T1 not being within period range
                cntTrerun = 0
                while self.warnT:
                    # Generally not being run at first iteration
                    if cntTrerun > 0:
                        rerun = True
                    else:
                        rerun = False

                    # Reruns
                    print("[RERUN] Rerun for fundamental period correction...")
                    gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(solution, omega, read=read)
                    modes_to_use = modes[direction] if self.flag3d else modes
                    phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes_to_use, table_sls,
                                                   rerun=rerun, direction=idx)
                    forces = next(phase_4)
                    demands, temp = next(phase_4)
                    # TODO, increasing cover results in issues, fix (maybe fixed with recent changes, verify)
                    details, new_hinge_model, hard_ductility, fract_ductility = next(phase_4)
                    warnMax, warnMin, warnings = next(phase_4)
                    if self.flag3d:
                        # Update the hinge models related to the direction of interest
                        hinge_models[tag] = new_hinge_model
                    else:
                        hinge_models = new_hinge_model

                    # Run modal analysis
                    model_periods, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                   t_upper, direction=direction)
                    # Update modes and period to cache
                    if self.flag3d:
                        modes[direction]["Periods"] = np.array([model_periods])
                        modes[direction]["Modes"][idx, :] = np.array(modalShape)
                        spoPattern = np.abs(modes[direction]["Modes"][idx, :])
                        periodCache = opt_sol[tag]["T"]
                    else:
                        modes["Periods"] = np.array(model_periods)
                        modes["Modes"][0, :] = np.array(modalShape)
                        spoPattern = np.abs(modalShape)
                        periodCache = opt_sol["T"]

                    cntTrerun += 1
                    if not self.warnT:
                        # Update optimal solution (the update difference should be minimal as significant changes
                        # not expected)
                        if self.flag3d:
                            opt_sol[tag]["Mstar"] = mstar
                            opt_sol[tag]["Part Factor"] = gamma
                            opt_sol[tag]["T"] = model_periods[0]
                        else:
                            opt_sol["Mstar"] = mstar
                            opt_sol["Part Factor"] = gamma
                            opt_sol["T"] = model_periods[0]
                        # TODO, rerun for new cy, detailing and MA. Or significant changes not expected?
                        print("[RERUN COMPLETE] Rerun for fundamental period correction")

                # Exiting warnT correction
                # Correction if unsatisfactory detailing, modifying only towards increasing c-s, based on phase 4
                if warnMax:
                    # Generally not being run at first iteration
                    """Look for a different solution"""
                    # Get the new design solution and the modal shapes
                    solution, m_temp = self.seek_solution(warnings, opt_sol, direction=direction)
                    print("[RERUN COMPLETE] New design solution has been selected due to unsatisfactory detailing...")
                    # Update the solutions
                    if self.flag3d:
                        opt_sol[tag] = solution
                        modes[direction]["Periods"] = m_temp["Periods"]
                        modes[direction]["Modes"] = m_temp["Modes"]
                    else:
                        opt_sol = solution
                        modes = m_temp

                """Create an OpenSees model and run static pushoveriterate analysis"""
                # For a single frame assumed yield base shear
                vy_assumed = cy * gamma * mstar * 9.81
                omegaCache = omega
                spoShapeCache = self.spo_shape
                spoResults, spo_idealized, omega = self.run_spo(opt_sol, hinge_models, forces, vy_assumed, spoPattern,
                                                                omega, direction=direction)

                # Record OpenSees outputs
                self.model_outputs = {"MA": {"T": model_periods, "modes": modalShape, "gamma": gamma, "mstar": mstar},
                                      "SPO": spoResults, "SPO_idealized": spo_idealized}

                # Reading SPO parameters from file (Set to False, as the actual shape is already identified)
                read = False

                print("[SUCCESS] Static pushover analysis was successfully performed.")

                # Print out information
                new_period = opt_sol[tag]["T"] if self.flag3d else opt_sol["T"]
                print("--------------------------")
                print(f"[ITERATION {cnt + 1} END] Actual over assumed values of variables are provided: \n"
                      f"Yield strength overstrength: {omega / omegaCache * 100:.0f}%, \n"
                      f"Hardening ductility: {self.spo_shape['mc'] / spoShapeCache['mc'] * 100:.0f}%, \n"
                      f"Fracturing ductility: {self.spo_shape['mf'] / spoShapeCache['mf'] * 100:.0f}%, \n"
                      f"Fundamental period: {new_period / periodCache:.0f}%.")
                print("--------------------------")

                # Increase count of iterations
                cnt += 1

            if cnt == maxiter:
                print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": gamma, "Mstar": mstar, "Period range": period_limits,
                         "overstrength": omega, "yield": [cy, dy], "muc": float(self.spo_shape["mc"])}

        return ipbsd_outputs, spo2ida_data, opt_sol, demands, details, hinge_models, forces, self.model_outputs
