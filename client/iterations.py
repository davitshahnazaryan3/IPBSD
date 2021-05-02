import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy import optimize
from external.crossSectionSpace import CrossSectionSpace
import sys


class Iterations:
    def __init__(self, ipbsd, sols, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath, gravity_loads, flag3d=False):
        """
        Initialize iterations
        :param ipbsd: object                            IPBSD object for input reading
        :param sols: dict                               Solutions containing structural element information
        :param spo_file: str                            Path to .csv containing SPO shape assumptions
        :param target_MAFC: float                       Target MAFC
        :param analysis_type: int                       Type of elastic analysis for design purpose
        :param damping: float                           Damping ratio
        :param num_modes: int                           Number of modes for RMSA
        :param fstiff: float                            Stiffness reduction factor unless hinge information is available
        :param rebar_cover: float                       Reinforcement cover
        :param outputPath: str                          Path to export outputs
        :param gravity_loads: dict                      Gravity loads to be applied
        :param flag3d: bool                             Whether 3D modelling is being carried out
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
        self.outputPath = outputPath
        self.gravity_loads = gravity_loads
        self.flag3d = flag3d
        # SPO shape
        self.spo_shape = None
        # Whether overstrength assumption is not correct (True necessitates iterations)
        self.omegaWarn = False
        # Whether initial secant to yield period is not correct (True necessitates iterations)
        self.warnT = True
        # Whether static pushover curve shape is not correct (False necessitates iterations)
        self.spo_validate = False
        # Model outputs to be exported
        self.model_outputs = None
        # Period to be used when performing important calculations
        self.period_to_use = None

    def compare_value(self, x, y, tol=0.05):
        """
        Verify whether x is within tolerance bounds of y
        :param x: float
        :param y: float
        :param tol: float
        :return: bool
        """
        if max(x, y) - tol * max(x, y) <= min(x, y) <= max(x, y) + tol * max(x, y):
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
        peak_x = x["a"] * (x["mc"] - 1) + 1
        peak_y = y["a"] * (y["mc"] - 1) + 1

        a1x = (1 + peak_x) / 2 * (x["mc"] - 1)
        a1y = (1 + peak_y) / 2 * (y["mc"] - 1)

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
        slope = m2 / d2

        # Fitting the plasticity portion based on the area under the curve
        y_pl = y[stfIdx: getIndex(Vmax, y)]
        nbins = len(y_pl) - 1
        dx = (dmax - d2) / nbins
        area_pl = np.trapz(y_pl, dx=dx)

        a = slope
        b = Vmax - slope * dmax
        c = 2 * area_pl - Vmax * dmax
        d = b ** 2 - 4 * a * c
        sol1 = (-b - np.sqrt(d)) / (2 * a)
        sol2 = (-b + np.sqrt(d)) / (2 * a)
        if sol1 > 0 and sol2 > 0:
            xint = min(sol1, sol2)
        elif sol1 > 0 and sol2 <= 0:
            xint = sol1
        else:
            xint = sol2

        yint = xint * slope

        # Determinant is negative, look for an alternative fitting approach
        # print("[WARNING SPO FITTING] Using an approximate method of fitting, as a solution is not feasible!")
        # f = lambda x: (0.5 * (Vmax + x[0]) * (dmax - x[0] / x[1]) - area_pl)
        # x0 = [m2, slope]
        # sol = optimize.least_squares(f, x0)
        # yint = min(sol.x[0], 0.85 * Vmax)
        # xint = yint / sol.x[1]

        # if d < 0:
        #     # Determinant is negative, look for an alternative fitting approach
        #     print("[WARNING SPO FITTING] Using an approximate method of fitting, as a solution is not feasible!")
        #     f = lambda x: (0.5 * (Vmax + x[0]) * (dmax - x[0] / x[1]) - area_pl)
        #     x0 = [m2, slope]
        #     sol = optimize.least_squares(f, x0)
        #     yint = min(sol.x[0], 0.85 * Vmax)
        #     xint = yint / sol.x[1]
        # else:
        #     # Force Vy not be larger than maximum V
        #     yint = min(yint, 0.99 * Vmax)

        # Find point of plasticity initiation
        stf0 = (y[1] - y[0]) / (x[1] - x[0])
        for i in range(1, len(x)):
            stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if stf1 <= 0.85 * stf0:
                break
            else:
                stf0 = stf1

        if i == getIndex(Vmax, y):
            i = i - 10

        dPl = x[i]
        mPl = y[i]

        a0, b0 = getEquation((d1, m1), (d2, m2))
        a1, b1 = getEquation((dPl, mPl), (dmax, Vmax))

        # Find intersection point, i.e. the nominal yield point
        xint = (b1 - b0) / (a0 - a1)
        yint = a0 * xint + b0

        # Now, identify the residual strength point (here defined at V=0)
        yres = max(y[-1], yint * residual)
        idx = getIndex(1.01 * yres, y[::-1])
        xres = x[::-1][idx]
        # Getting the actual residual strength and corresponding displacement
        ymin = y[-1]
        # xmin = (Vmax - ymin) * (xres - dmax) / (Vmax - yres) + dmax

        # Select the softening slope until residual displacement
        # Fitting based on the area under the softening slope
        y_soft = y[getIndex(Vmax, y): getIndex(xres, x)]
        nbins = len(y_soft) - 1
        dx = (xres - dmax) / nbins
        area_soft = np.trapz(y_soft, dx=dx)
        xmin = 2 * area_soft / (Vmax + ymin) + dmax

        # Avoid negative residual strength and zero as residual strength
        if ymin <= 0:
            ymin = 10.

        # Get the curve
        d = np.array([0., xint, dmax, xmin])
        v = np.array([0., yint, Vmax, ymin])

        return d, v

    def derive_spo(self, spo, residual=0.1):
        # Top displacement and base shear
        try:
            x = spo["d"]
            y = spo["v"]
        except:
            x = spo[0]
            y = spo[1]

        def getIndex(x, data):
            if np.where(data >= x)[0].size == 0:
                return np.nan
            else:
                return np.where(data >= x)[0][0]

        def getEquation(p1, p2):
            points = [p1, p2]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            return m, c

        """
        The point below the max point is quite subjective
        So we need to identify two points
        Keep Vmax as the max point as long as the 
        Stiffness is reducing consistently
        And as long as the V is not varying from the max
        significantly.
        """
        # Identify the maximum point
        ymax = max(y)
        xmax = x[getIndex(ymax, y)]

        # Gradient of y
        grad = np.gradient(y)

        # Look for a very steep gradient
        idx = getIndex(10, -grad) - 1
        # Make sure that the new potential peak is not way lower than the maximum value
        # This new peak is due to P-delta effects
        if y[idx] / ymax >= 0.85:
            ymax = y[idx]
            xmax = x[idx]

        # Yield point
        # Get initial stiffness
        m1 = 0.2 * ymax
        d1 = x[getIndex(m1, y)]
        stiff_elastic = m1 / d1
        temp = y / x
        stfIdx = np.where(temp < 0.9 * stiff_elastic)[0][0]
        d2 = x[stfIdx]
        m2 = y[stfIdx]
        slope = m2 / d2

        # Fitting the plasticity portion based on the area under the curve
        y_pl = y[stfIdx: getIndex(ymax, y)]
        nbins = len(y_pl) - 1
        dx = (xmax - d2) / nbins
        area_pl = np.trapz(y_pl, dx=dx)

        a = slope
        b = ymax - slope * xmax
        c = 2 * area_pl - ymax * xmax
        d = b ** 2 - 4 * a * c
        sol1 = (-b - np.sqrt(d)) / (2 * a)
        sol2 = (-b + np.sqrt(d)) / (2 * a)
        if sol1 > 0 and sol2 > 0:
            xint = min(sol1, sol2)
        elif sol1 > 0 and sol2 <= 0:
            xint = sol1
        else:
            xint = sol2

        yint = xint * slope

        # Find point of plasticity initiation
        stf0 = (y[1] - y[0]) / (x[1] - x[0])
        for i in range(1, len(x)):
            stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if stf1 <= 0.85 * stf0:
                break
            else:
                stf0 = stf1

        if i == getIndex(ymax, y):
            i = i - 10

        dPl = x[i]
        mPl = y[i]

        a0, b0 = getEquation((d1, m1), (d2, m2))
        a1, b1 = getEquation((dPl, mPl), (xmax, ymax))

        # Find intersection point, i.e. the nominal yield point
        xint = (b1 - b0) / (a0 - a1)
        yint = a0 * xint + b0

        # Now, identify the residual strength point (here defined at V=0)
        yres = max(y[-1], yint * residual)
        idx = getIndex(1.01 * yres, y[::-1])
        xres = x[::-1][idx]
        # Getting the actual residual strength and corresponding displacement
        ymin = y[-1]
        # xmin = (Vmax - ymin) * (xres - dmax) / (Vmax - yres) + dmax

        # Select the softening slope until residual displacement
        # Fitting based on the area under the softening slope
        y_soft = y[getIndex(ymax, y): getIndex(xres, x)]
        nbins = len(y_soft) - 1
        dx = (xres - xmax) / nbins
        area_soft = np.trapz(y_soft, dx=dx)
        xmin = 2 * area_soft / (ymax + ymin) + xmax

        # Avoid negative residual strength and zero as residual strength
        if ymin <= 0:
            ymin = 10.
        # Make sure that peak is not lower than yield point (incompatible for SPO2IDA)
        if yint > ymax:
            ymax = yint

        # Get the curve
        d = np.array([0., xint, xmax, xmin])
        v = np.array([0., yint, ymax, ymin])

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
        # Get the seismic solution of interest
        if self.flag3d:
            opt = opt_sol[direction + "_seismic"]
        else:
            opt = opt_sol

        # Number of storeys
        nst = self.ipbsd.data.nst

        # Remove column values not related to cross-section dimensions
        if self.ipbsd.data.configuration == "perimeter":
            cols_to_drop = ["T", "Weight", "Mstar", "Part Factor"]
            opt.loc[cols_to_drop] = np.nan

        # All solutions
        if self.ipbsd.data.configuration == "perimeter":
            all_solutions = self.sols[direction]
        else:
            # For space systems solutions are more entangled
            all_solutions = self.sols

        # Create an empty dictionary of warnings
        w = {x: False for x in opt.keys()}

        # After modifications, equally between c-s of two storeys might not hold
        # Increment for cross-section modifications for elements with warnings
        increment = 0.05
        any_warnings = False
        # Columns
        for ele in warnings["MAX"]["Columns"]:
            if warnings["MAX"]["Columns"][ele] == 1:
                any_warnings = True

                # Modify cross-section
                storey = ele[1]
                bay = int(ele[3])
                if bay == 1:
                    opt[f"he{storey}"] = opt[f"he{storey}"] + increment
                    w[f"he{storey}"] = True
                else:
                    opt[f"hi{storey}"] = opt[f"hi{storey}"] + increment
                    w[f"hi{storey}"] = True

        # Beams
        for i in warnings["MAX"]["Beams"]:
            for ele in warnings["MAX"]["Beams"][i]:
                storey = ele[1]
                if warnings["MAX"]["Beams"][i][ele] == 1:
                    any_warnings = True

                    # Increase section cross-section
                    opt[f"b{storey}"] = opt[f"he{storey}"]
                    opt[f"h{storey}"] = opt[f"h{storey}"] + increment
                    w[f"h{storey}"] = True

        # If any section was modified, we need to reapply the constraints
        perp_dir = None
        opt_perp = None
        opt_gr = None
        if self.ipbsd.data.configuration == "space":
            # Modify the seismic solution of dependencies of perpendicular direction
            perp_dir = "y" if direction == "x" else "x"
            opt_perp = opt_sol[perp_dir + "_seismic"]
            opt_gr = opt_sol["gravity"]

            # Call CrossSectionSpace object to fix dependencies
            fd = CrossSectionSpace(self.ipbsd.data, None, None)
            for key in w.keys():
                # Check whether warning was raised (i.e. component cross-section was modified)
                if w[key]:
                    # Call to fix dependencies
                    opt, opt_perp, opt_gr = fd.fix_dependencies(key, opt, opt_perp, opt_gr)

        else:
            if any_warnings:
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
                        opt[f"b{st}"] = opt[f"h{st}"] - 0.3

        # Finding a matching solution from the already generated DataFrame
        if self.ipbsd.data.configuration == "perimeter":
            solution = all_solutions[all_solutions == opt].dropna(thresh=len(all_solutions.columns) - 4)
            solution = all_solutions.loc[solution.index]

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
        else:
            # Space systems
            opt_sol[direction + "_seismic"] = opt
            opt_sol[perp_dir + "_seismic"] = opt_perp
            opt_sol["gravity"] = opt_gr

            return opt_sol

    def iterate_phase_3(self, opt_sol, omega, read=True):
        """
        Runs phase 3 of the framework
        :param opt_sol: Series                          Solution containing modal period
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
            # Initial secant to yield period from SPO analysis
            period = self.period_to_use
        else:
            period = round(float(opt_sol['T']), 1)

        if read:
            # Reads the input assumption if necessary
            self.spo_shape = self.ipbsd.data.initial_spo_data(period, self.spo_file)

        # Modify period of spo shape to period
        self.spo_shape["T"] = period

        # Run SPO2IDA
        spo2ida_data = self.ipbsd.perform_spo2ida(self.spo_shape)
        print("[SUCCESS] SPO2IDA was performed")

        """Yield strength optimization for MAFC and verification"""
        part_factor = opt_sol["Part Factor"]
        m_star = opt_sol["Mstar"]
        cy, dy = self.ipbsd.verify_mafc(period, spo2ida_data, part_factor, self.target_MAFC, omega, hazard="True")
        print("[SUCCESS] MAFC was validated")
        return part_factor, m_star, cy, dy, spo2ida_data

    def iterate_phase_4(self, cy, dy, sa, period_range, solution, modes, table_sls, rerun=False, direction=0,
                        gravity_demands=None, detail_gravity=True):
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
        :param gravity_demands: dict                    Demands on central/gravity structural elements
        :param detail_gravity: bool                     Perform detailing of central/gravity structural elements
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
        # NOTE: RMSA not supported yet
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

        yield forces

        """Perform ELF and/or gravity analysis"""
        def analyze(hinge=None):
            # Get the system configuration
            system = self.ipbsd.data.configuration.lower()

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
                                                  hinge=hinge, direction=direction, system=system)
            elif self.analysis_type == 3:
                demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                                  fstiff=self.fstiff, hinge=hinge, direction=direction, system=system)

            elif self.analysis_type == 4 or self.analysis_type == 5:
                demands = {}
                for mode in range(self.num_modes):
                    demands[f"Mode{mode + 1}"] = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                                         list(forces["Fi"][:, mode]),
                                                                         fstiff=self.fstiff, hinge=hinge,
                                                                         direction=direction, system=system)

                demands = self.ipbsd.perform_cqc(corr, demands)

                if self.analysis_type == 5:
                    demands_gravity = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                              grav_loads=list(forces["G"]), system=system,
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

        # Demands on all elements of the system where plastic hinge information is missing
        demands = analyze(hinge=None)
        # Update the demands on central/gravity elements
        if self.flag3d:
            if direction == 0:
                gravity_demands["x"] = demands["gravity"]
            else:
                gravity_demands["y"] = demands["gravity"]
        print("[SUCCESS] Analysis completed and demands on structural elements were estimated.")

        """Design the structural elements"""
        # Details the frame of seismic direction only
        if self.flag3d:
            seismic_demands = demands["x_seismic"] if direction == 0 else demands["y_seismic"]
        else:
            seismic_demands = demands
        details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
            self.ipbsd.design_elements(seismic_demands, seismic_solution, modes, dy, cover=self.rebar_cover,
                                       direction=direction)

        # Design elements of central/gravity elements if space systems are used - since they are part of the lateral
        # force resisting system
        if detail_gravity and self.flag3d:
            hinge_gravity = self.ipbsd.design_elements(gravity_demands, solution["gravity"], None, None,
                                                       cover=self.rebar_cover, direction=direction, gravity=True)
        else:
            hinge_gravity = None

        print("[SUCCESS] Section detailing done. Element idealized Moment-Curvature relationships obtained")

        # Rerun elastic analysis with updated stiffness reduction factors for all structural elements
        if rerun:
            if self.flag3d:
                # Store the results into hinge and rerun analysis with updated stiffness
                if direction == 0:
                    hinge["x_seismic"] = hinge_models
                else:
                    hinge["y_seismic"] = hinge_models
                if detail_gravity:
                    hinge["gravity"] = hinge_gravity
                demands = analyze(hinge=hinge)
            else:
                demands = analyze(hinge=hinge_models)

            details, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
                self.ipbsd.design_elements(seismic_demands, solution, modes, dy, cover=self.rebar_cover,
                                           direction=direction)
            # Design elements of central/gravity elements if space systems are used
            if detail_gravity and self.flag3d:
                hinge_gravity = self.ipbsd.design_elements(gravity_demands, solution["gravity"], None, None,
                                                           cover=self.rebar_cover, direction=direction, gravity=True)

            print("[RERUN COMPLETE] Rerun of analysis and detailing complete due to modified stiffness.")

        if self.flag3d:
            if direction == 0:
                hinge["x_seismic"] = hinge_models
            else:
                hinge["y_seismic"] = hinge_models
            if detail_gravity:
                hinge["gravity"] = hinge_gravity
            demands_gravity = demands["gravity"]
        else:
            demands_gravity = None

        yield demands, demands_gravity
        yield details, hinge_models, hinge_gravity, mu_c, mu_f
        yield warnMax, warnMin, warnings

    def run_ma(self, opt_sol, hinge, forces, t_upper, tol=1.05, direction="x", spo_period=None, do_corrections=True):
        """
        Creates a nonlinear model and runs Modal Analysis with the aim of correcting opt_sol and fundamental period
        :param opt_sol: DataFrame                   Optimal solution
        :param hinge: DataFrame                     Hinge models after detailing
        :param forces: DataFrame                    Acting forces (gravity, lateral)
        :param t_upper: float                       Upper period limit from IPBSD
        :param tol: float                           Tolerance of upper period limit satisfaction
        :param direction: str                       Direction of action (x for 2D, x or y for 3D)
        :param spo_period: float                    SPO based period
        :param do_corrections: bool                 Make corrections if period condition is not met (typically at True)
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
            else:
                if opt_sol["x_seismic"]["T"] >= opt_sol["y_seismic"]["T"]:
                    idx = 1
                else:
                    idx = 0
            seismic_solution = opt_sol[f"{direction}_seismic"]
            gamma = gamma[idx]
            mstar = mstar[idx]
            model_period = model_periods[idx]
            if self.ipbsd.data.configuration == "perimeter":
                all_solutions = self.sols[direction]
            else:
                all_solutions = self.sols
        else:
            # Interested only in first mode for the 2D approach
            model_period = model_periods[0]
            seismic_solution = opt_sol
            all_solutions = self.sols
            idx = 0

        # We use the secant period to yield for the validation
        if spo_period is not None:
            model_period = spo_period

        # There is a high likelihood that Fundamental period and SPO curve shape will not match the assumptions
        # Therefore the first iteration should correct both assumptions (further corrections are likely not to be large)
        if model_period > tol * t_upper and do_corrections:
            if direction == "y" and self.ipbsd.data.configuration == "space":
                # Modifications are made to the sections associated with y direction only
                # Currently a new solution is being sought increasing c-s dimensions by 0.5
                for st in range(self.ipbsd.data.nst):
                    opt_sol["y_seismic"][f"hi{st+1}"] += 0.05
                    opt_sol["y_seismic"][f"h{st+1}"] += 0.05
                    opt_sol["gravity"][f"hy{st+1}"] += 0.05
                    # If any of the cross-section dimensions is beyond an undesirable value, raise a warning
                    vlist = [opt_sol["y_seismic"][f"hi{st+1}"], opt_sol["y_seismic"][f"h{st+1}"],
                             opt_sol["gravity"][f"hy{st+1}"]]
                    if not all(v < 0.95 for v in vlist):
                        print("[WARNING] Cross-section dimensions are above 0.9m.")

                # Actual period of the structure is guessed
                self.period_to_use = t_upper
                self.warnT = True
            else:
                tag = "T" if self.ipbsd.data.configuration == "perimeter" else f"T{idx + 1}"
                # # Get index of the seismic solution
                # idx_seismic = seismic_solution.name
                # # Get the initial period assumption corresponding to the index from the solutions database
                # period_initial = all_solutions.loc[idx_seismic][tag]
                # Period error (compared to the period from the all solutions variable)
                period_error = model_period - seismic_solution["T"]
                # Look for a new period for the design solution (probably not the best way to guess)
                tnew = t_upper - period_error
                # Select all solutions in the vicinity of the new period
                sols = all_solutions[(all_solutions[tag] >= tnew - tnew * (tol - 1)) &
                                     (all_solutions[tag] <= tnew + tnew * (tol - 1))]
                # Select the solution with the least weight
                seismic_solution = sols[sols["Weight"] == sols["Weight"].min()].iloc[0]
                if self.ipbsd.data.configuration == "perimeter":
                    # For perimeter systems
                    seismic_solution = seismic_solution[f"{direction}_seismic"]
                # Actual period of the structure is guessed to be (opt_sol period + error)
                self.period_to_use = seismic_solution[tag] + period_error
                self.warnT = True

                if self.flag3d and self.ipbsd.data.configuration == "perimeter":
                    # Update the optimal solution
                    opt_sol[f"{direction}_seismic"] = seismic_solution
                else:
                    # For 2D systems and space systems
                    if self.ipbsd.data.configuration == "space":
                        cs = CrossSectionSpace(self.ipbsd.data, None, None)
                        seismic_solution = cs.get_section(seismic_solution)
                    opt_sol = seismic_solution

        else:
            # If the model period was within the confidence
            self.period_to_use = model_period
            self.warnT = False

        # # Update period of opt_sol if it is not matching the actual one
        # if self.flag3d:
        #     if model_periods != opt_sol[direction + "_seismic"]["T"]:
        #         opt_sol[direction + "_seismic"]["T"] = model_periods
        #         # Update the remaining modal properties
        #         opt_sol[direction + "_seismic"]["Part Factor"] = gamma
        #         opt_sol[direction + "_seismic"]["Mstar"] = mstar
        # else:
        #     if model_periods != opt_sol["T"]:
        #         opt_sol["T"] = model_periods
        #         # Update the remaining modal properties
        #         opt_sol["Part Factor"] = gamma
        #         opt_sol["Mstar"] = mstar

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
        # DEVELOPER TOOL
        new_fitting = False
        if new_fitting:
            d, v = self.derive_spo(spoResults, residual=0.3)
        else:
            d, v = self.derive_spo_shape(spoResults, residual=0.3)

        # Actual overstrength
        if self.flag3d and self.ipbsd.data.configuration == "perimeter":
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
        self.spo_validate = self.compare_areas(self.spo_shape, spo_data_updated, tol=0.2)

        # Update the SPO parameters
        if not self.spo_validate:
            self.spo_shape = spo_data_updated
        else:
            self.spo_shape = self.spo_shape

        return spoResults, [d, v], omegaNew

    def generate_initial_solutions(self, opt_sol, opt_modes, omega, sa, period_range, table_sls):
        """
        Master file to run for each direction/iteration (only for 3D modelling)
        :param opt_sol: dict                        Dictionary containing structural element cross-section information
        :param opt_modes: dict                      Modal properties of the solution
        :param omega: float                         Overstrength factor
        :param sa: array                            Spectral accelerations for spectrum at SLS
        :param period_range: array                  Period range for spectrum at SLS
        :param table_sls: DataFrame                 DBD table at SLS
        :return: Design outputs and Demands on gravity (internal elements)
        """
        # Call the iterations function (iterations have not yet started though)
        # For 3D option
        frames = ["x", "y"]
        # Initialize dictionary for storing design outputs
        design_outputs = {"x": {}, "y": {}, "gravity": {}}
        # Initialize demands on central/gravity structural elements
        bx_gravity = np.zeros((self.ipbsd.data.nst, self.ipbsd.data.n_bays, len(self.ipbsd.data.spans_y) - 1))
        by_gravity = np.zeros((self.ipbsd.data.nst, self.ipbsd.data.n_bays - 1, len(self.ipbsd.data.spans_y)))
        c_gravity = np.zeros((self.ipbsd.data.nst, self.ipbsd.data.n_bays - 1, len(self.ipbsd.data.spans_y) - 1))
        gravity_temp = {"Beams_x": {"M": {"Pos": bx_gravity.copy(), "Neg": bx_gravity.copy()},
                                    "N": bx_gravity.copy(), "V": bx_gravity.copy()},
                        "Beams_y": {"M": {"Pos": by_gravity.copy(), "Neg": by_gravity.copy()},
                                    "N": by_gravity.copy(), "V": by_gravity.copy()},
                        "Columns": {"M": c_gravity.copy(), "N": c_gravity.copy(), "V": c_gravity.copy()}}
        demands_gravity = {"x": gravity_temp, "y": gravity_temp}

        # For each direction
        for i in frames:

            if self.ipbsd.data.configuration == "space":
                modes = opt_modes
            else:
                modes = opt_modes[i]

            print(f"[PHASE] Getting initial solution in {i} direction.")
            # Get seismic solution in the direction of interest
            solution = opt_sol[f"{i}_seismic"]
            table = table_sls[i]
            direction = 0 if i == "x" else 1

            if i == "x":
                # Applying lateral loads in X direction
                print("[INITIATE] Designing frame in X direction!")
            else:
                # Applying lateral loads in Y direction
                print("[INITIATE] Designing frame in Y direction!")

            # Initialize period to use (reset when running for Y direction)
            self.period_to_use = None

            # Generate initial solutions for both directions
            # For each primary direction of building, apply lateral loads and design the structural elements
            print("[PHASE] Commencing phase 3...")
            gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(solution, omega)

            """Get action and demands"""
            print("[PHASE] Commencing phase 4...")
            phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table, direction=direction,
                                           gravity_demands=demands_gravity, detail_gravity=False)
            forces = next(phase_4)
            demands, demands_gravity[i] = next(phase_4)
            details, hinge_models, hinge_gravity, hard_ductility, fract_ductility = next(phase_4)
            warnMax, warnMin, warnings = next(phase_4)

            # Append information to design solutions
            design_outputs[i]["forces"] = forces
            design_outputs[i]["demands"] = demands
            design_outputs[i]["details"] = details
            design_outputs[i]["hinge_models"] = hinge_models
            design_outputs[i]["warnings"] = {"warnMax": warnMax, "warnMin": warnMin, "warnings": warnings}
            design_outputs[i]["phase3"] = {"cy": cy, "dy": dy, "spo2ida_data": spo2ida_data, "gamma": gamma,
                                           "mstar": mstar}

        # Modify hinge elements of external seismic columns to the strongest (larger My) from designs of both directions
        hinge_models_x = design_outputs["x"]["hinge_models"]
        hinge_models_y = design_outputs["y"]["hinge_models"]
        external_hinges_x = hinge_models_x[(hinge_models_x["Position"] == "external") &
                                           (hinge_models_x["Element"] == "Column")].reset_index()
        external_hinges_y = hinge_models_y[(hinge_models_y["Position"] == "external") &
                                           (hinge_models_y["Element"] == "Column")].reset_index()

        for index, row in external_hinges_x.iterrows():
            my_x = external_hinges_x["m1"].iloc[index]
            my_y = external_hinges_y["m1"].iloc[index]
            idx_x = external_hinges_x["index"].iloc[index]
            idx_y = external_hinges_y["index"].iloc[index]
            bay_n_x = external_hinges_x["Bay"].iloc[index]
            bay_n_y = external_hinges_y["Bay"].iloc[index]

            if my_x >= my_y:
                hinge_models_y.iloc[idx_y] = external_hinges_x.drop(columns=["index"]).iloc[index]
                # Modify corresponding Bay number
                hinge_models_y.at[idx_y, "Bay"] = bay_n_y

            else:
                hinge_models_x.iloc[idx_x] = external_hinges_y.drop(columns=["index"]).iloc[index]
                hinge_models_x.at[idx_x, "Bay"] = bay_n_x

        design_outputs["x"]["hinge_models"] = hinge_models_x
        design_outputs["y"]["hinge_models"] = hinge_models_y

        # TODO, add warnings for central elements of space systems as well
        # Detail the gravity frames in both direction (envelope)
        hinge_gravity = self.ipbsd.design_elements(demands_gravity, opt_sol["gravity"], None, None,
                                                   cover=self.rebar_cover, direction=0, gravity=True)
        design_outputs["gravity"]["hinge_models"] = hinge_gravity

        return design_outputs, demands_gravity

    def run_iterations_for_3d(self, design_outputs, demands_gravity, period_limits, opt_sol, opt_modes, sa,
                              period_range, table_sls, iterate=True, maxiter=20, omega=None):
        """
        Runs iterations for 3D approach in both directions of action
        :param design_outputs: dict
        :param demands_gravity: dict
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
            print(f"[INITIATE 3D] Running framework in {i} direction!")
            table = table_sls[i]
            periods = period_limits[i]

            ipbsd_outputs, spo2ida_data, opt_sol, modes, demands, details, hinge_models, forces, model_outputs, \
            demands_gravity = \
                self.validations(opt_sol, opt_modes, sa, period_range, table, periods, iterate=iterate, maxiter=maxiter,
                                 omega=omega, direction=i, initial_design_sols=design_outputs,
                                 demands_gravity=demands_gravity)

            # Update parameters to be used in Y direction
            if i == "x":
                # Modal participation factor and effective modal mass
                design_outputs["y"]["phase3"]["gamma"] = opt_sol["y_seismic"]["Part Factor"]
                design_outputs["y"]["phase3"]["mstar"] = opt_sol["y_seismic"]["Mstar"]
                # Update optimal modes
                opt_modes["Periods"] = modes["x"]["Periods"]
                opt_modes["Modes"] = modes["x"]["Modes"].transpose()
                # Update hinge models in initial_design_sols
                design_outputs["x"]["hinge_models"] = hinge_models["x_seismic"]
                design_outputs["gravity"]["hinge_models"] = hinge_models["gravity"]

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
                    omega=None, direction="x", initial_design_sols=None, demands_gravity=None):
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
        :param demands_gravity: dict                    Demands on central/gravity structural elements
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
            details, hinge_models, hinge_gravity, hard_ductility, fract_ductility = next(phase_4)
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

        # Number of seismic frames
        if self.flag3d and self.ipbsd.data.configuration == "perimeter":
            # For perimeter seismic frames
            seismic_frames = 2
        else:
            # For 2D frames and space systems
            seismic_frames = 1

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
            spo_period = None

            # Correction for period
            """Create an OpenSees model and run modal analysis.
            Acts as an initial estimation of modal properties."""
            if self.ipbsd.data.configuration == "perimeter" and direction == "x":
                # For Space systems it has already been run
                model_period, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces, t_upper,
                                                                              direction=direction)
            else:
                idx = 0 if direction == "x" else 1
                model_period = modes["Periods"]
                modalShape = modes["Modes"]
                modes = {"x": {"Periods": modes["Periods"], "Modes": modes["Modes"].transpose()},
                         "y": {"Periods": modes["Periods"], "Modes": modes["Modes"].transpose()}}
            
            # Reread initial assumption of pushover shape (Same as for X direction)
            if direction == "y":
                # Reread initial assumption of pushover shape (Same as for X direction)
                self.spo_shape = self.ipbsd.data.initial_spo_data(self.period_to_use, self.spo_file)
                modes_to_use = modes[direction]

                gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(opt_sol["y_seismic"], omega)
                phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes_to_use, table_sls,
                                               gravity_demands=demands_gravity, direction=1)
                forces = next(phase_4)
                demands, dem_gr = next(phase_4)
                details, new_hinge_model, hinge_gravity, hard_ductility, fract_ductility = next(phase_4)
                warnMax, warnMin, warnings = next(phase_4)
                # Update the gravity demands
                demands_gravity[direction] = dem_gr

                # Update the hinge models
                hinge_models["y_seismic"] = new_hinge_model
                hinge_models["gravity"] = hinge_gravity

                if self.ipbsd.data.configuration == "space":
                    model_period, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                  t_upper, direction=direction)

            # Update modes and some utility variables
            if self.flag3d:
                idx = 0 if direction == "x" else 1
                # Modal properties
                modes["x"]["Periods"] = model_period
                modes["x"]["Modes"] = modalShape.transpose()
                modes["y"]["Periods"] = model_period
                modes["y"]["Modes"] = modalShape.transpose()
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
                modes["Periods"] = np.array(model_period)
                modes["Modes"][0, :] = np.array(modalShape)
                spoPattern = np.abs(modalShape)
                solution = opt_sol
                tag = None
                periodCache = opt_sol["T"]

            # Iterate until all conditions are met
            while (self.warnT or warnMax or not self.spo_validate or self.omegaWarn) and cnt + 1 <= maxiter:

                # Iterations related to SPO corrections (skip first iteration, before running SPO)
                if (not self.spo_validate or self.omegaWarn or self.warnT) and cnt > 0:
                    # Not being run at first iteration, however almost always needs to be run at second iteration, since
                    # guessing SPO shape is nearly impossible
                    # Rerun Modal analysis if warnT was raised
                    if self.warnT:
                        # Now, we want to ensure that the new optimal solution will result in a better period estimate
                        model_period, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                      t_upper, direction=direction,
                                                                                      spo_period=spo_period)

                    # Reruns
                    print("[RERUN] Rerun for SPO shape correction...")
                    if self.flag3d:
                        solution = opt_sol[direction + "_seismic"]
                    else:
                        solution = opt_sol

                    # Calculates the new cy for the corrected SPO shape, period, Overstrength and c-s
                    gamma, mstar, cy, dy, spo2ida_data = self.iterate_phase_3(solution, omega, read=read)

                    # Run elastic analysis and detail the structure
                    # Update modal shape (phi) in table? This is used to identify the acting lateral forces when
                    # designing. If updated the period limits may change, so, I think at the end of the day we just
                    # want to compare how much they vary compared to the initial assumption as for each solution they
                    # will be different. For now, let's try updating it, so that the new lateral loads are calculated
                    # based on the updated shape without modifying the period limits
                    # for st in range(len(modalShape)):
                    #     table_sls[f"{st + 1}"]["phi"] = modalShape[st]
                    # TODO, issue when designing and detailing in momentcurvaturerc

                    modes_to_use = modes[direction] if self.flag3d else modes

                    phase_4 = self.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes_to_use, table_sls,
                                                   gravity_demands=demands_gravity, direction=idx)

                    forces = next(phase_4)
                    demands, dem_gr = next(phase_4)
                    details, new_hinge_model, hinge_gravity, hard_ductility, fract_ductility = next(phase_4)
                    warnMax, warnMin, warnings = next(phase_4)

                    # Update the gravity demands
                    demands_gravity[direction] = dem_gr

                    if self.flag3d:
                        # Update the hinge models related to the direction of interest
                        hinge_models[tag] = new_hinge_model
                        # Update the hinge models related to central/gravity structural elements
                        hinge_models["gravity"] = hinge_gravity
                    else:
                        hinge_models = new_hinge_model

                    # Run modal analysis to check the T1 if there was no period warning when running SPO
                    if not self.warnT:
                        model_period, modalShape, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, forces,
                                                                                      t_upper, direction=direction,
                                                                                      spo_period=spo_period)

                    # Update modes
                    if self.flag3d:
                        modes["x"]["Periods"] = model_period
                        modes["x"]["Modes"] = modalShape.transpose()
                        modes["y"]["Periods"] = model_period
                        modes["y"]["Modes"] = modalShape.transpose()
                        spoPattern = np.abs(modes[direction]["Modes"][idx, :])
                    else:
                        modes["Periods"] = np.array(model_period)
                        modes["Modes"][0, :] = np.array(modalShape)
                        spoPattern = np.abs(modalShape)

                    print("[RERUN COMPLETE] Rerun for SPO shape correction.")

                # Exiting warnT correction
                # Correction if unsatisfactory detailing, modifying only towards increasing c-s, based on phase 4
                # TODO: Warnings on gravity/central columns to be added
                if warnMax:
                    # Generally not being run at first iteration
                    """Look for a different solution"""
                    # Get the new design solution and the modal shapes
                    # Update the solutions
                    if self.ipbsd.data.configuration == "perimeter":
                        solution, m_temp = self.seek_solution(warnings, opt_sol, direction=direction)
                        if self.flag3d:
                            opt_sol[tag] = solution
                            modes[direction]["Periods"] = m_temp["Periods"]
                            modes[direction]["Modes"] = m_temp["Modes"]
                        else:
                            opt_sol = solution
                            modes = m_temp
                    else:
                        opt_sol = self.seek_solution(warnings, opt_sol, direction=direction)

                    print("[RERUN COMPLETE] New design solution was selected due to unsatisfactory detailing...")

                """Create an OpenSees model and run static pushover - iterate analysis.
                Acts as the first estimation of secant to yield period. Second step after initial modal analysis.
                Here, the secant to yield period might vary from initial modal period. 
                Therefore, we need to use the former one."""
                # For a single frame assumed yield base shear
                vy_assumed = cy * gamma * mstar * 9.81
                omegaCache = omega
                spoShapeCache = self.spo_shape
                spoPattern = np.round(spoPattern, 2)

                spoResults, spo_idealized, omega = self.run_spo(opt_sol, hinge_models, forces, vy_assumed, spoPattern,
                                                                omega, direction=direction)

                # Recalculate period as secant to yield period
                spo_period = 2 * np.pi * np.sqrt(mstar * seismic_frames / (spo_idealized[1][1] / spo_idealized[0][1]))
                self.period_to_use = spo_period

                if not spo_period <= t_upper * 1.05:
                    # If SPO based secant to yield period is not within the tolerance of upper period limit
                    self.warnT = True
                else:
                    # Even though SPO period might not match modal period, the condition is still satisfying
                    self.warnT = False

                # Record OpenSees outputs
                self.model_outputs = {"MA": {"T": spo_period, "modes": modalShape, "gamma": gamma, "mstar": mstar},
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
                      f"Fundamental period: {spo_period / periodCache:.0f}%.")
                print("--------------------------")

                # Increase count of iterations
                cnt += 1

            if cnt == maxiter:
                print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": gamma, "Mstar": mstar, "Period range": period_limits,
                         "overstrength": omega, "yield": [cy, dy], "muc": float(self.spo_shape["mc"]),
                         "spo2ida": self.spo_shape}

        return ipbsd_outputs, spo2ida_data, opt_sol, modes, demands, details, hinge_models, forces, \
               self.model_outputs, demands_gravity
