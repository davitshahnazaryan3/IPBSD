import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy import optimize
from external.crossSectionSpace import CrossSectionSpace
import sys


def geo_mean(iterable):
    a = np.log(iterable)
    return np.exp(a.mean())


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


class SeekDesign:
    """For 3D modelling only"""

    def __init__(self, ipbsd, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath, gravity_loads):
        """
        Initialize iterations
        :param ipbsd: object                            IPBSD object for input reading
        :param spo_file: str                            Path to .csv containing SPO shape assumptions
        :param target_MAFC: float                       Target MAFC
        :param analysis_type: int                       Type of elastic analysis for design purpose
        :param damping: float                           Damping ratio
        :param num_modes: int                           Number of modes for RMSA
        :param fstiff: float                            Stiffness reduction factor unless hinge information is available
        :param rebar_cover: float                       Reinforcement cover
        :param outputPath: str                          Path to export outputs
        :param gravity_loads: dict                      Gravity loads to be applied
        """
        self.ipbsd = ipbsd
        self.spo_file = spo_file
        self.target_MAFC = target_MAFC
        self.analysis_type = analysis_type
        self.damping = damping
        self.num_modes = num_modes
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.outputPath = outputPath
        self.gravity_loads = gravity_loads
        # SPO shape
        self.spo_shape = {}
        # Whether overstrength assumption is not correct (True necessitates iterations)
        self.omegaWarn = False
        # Whether initial secant to yield period is not correct (True necessitates iterations)
        self.warnT = True
        # Whether static pushover curve shape is not correct (False necessitates iterations)
        self.spo_validate = False
        # Model outputs to be exported
        self.model_outputs = None
        # Period to be used when performing important calculations (list)
        self.period_to_use = None
        # Initialize detailing results
        self.details = None

    def target_for_mafc(self, solution, omega, read=True):
        """
        Look for a spectral acceleration at yield to target for MAFC
        :param solution: dict               Cross-section information of the building's structural components
        :param omega: float                 Overstrength factor
        :param read: bool                   Read from file?
        :return cy: float                   Maximum Yield spectral acceleration in g (max of both directions)
        :return dy_xy: list                 Yield displacements in both directions
        :return spo2ida_data: dict          SPO2IDA results
        """
        # Overstrength
        if not isinstance(omega, list):
            overstrength = [omega] * 2
        else:
            overstrength = omega

        # Set period equal to the actual period computed from MA or SPO analysis
        if self.period_to_use is not None:
            # Initial secant to yield period from SPO analysis
            period = self.period_to_use
        else:
            Tx = round(float(solution["x_seismic"]['T']), 1)
            Ty = round(float(solution["y_seismic"]['T']), 1)
            period = [Tx, Ty]

        if read:
            # Reads the input assumption for SPO2IDA from file, if necessary
            self.spo_shape["x"] = self.ipbsd.data.initial_spo_data(period[0], self.spo_file)
            self.spo_shape["y"] = self.ipbsd.data.initial_spo_data(period[1], self.spo_file)

        # Update the period in case read was False
        self.spo_shape["x"]["T"] = period[0]
        self.spo_shape["y"]["T"] = period[1]

        """Yield strength optimization for MAFC and verification"""
        part_factor = [solution["x_seismic"]["Part Factor"], solution["y_seismic"]["Part Factor"]]

        # Run SPO2IDA
        cy_xy = []
        dy_xy = []
        spo2ida_data = {}
        for key in self.spo_shape.keys():
            i = 0 if key == "x" else 1
            spo2ida_data[key] = self.ipbsd.perform_spo2ida(self.spo_shape[key])
            cy, dy = self.ipbsd.verify_mafc(period[i], spo2ida_data[key], part_factor[i], self.target_MAFC,
                                            overstrength[i], hazard="True")
            cy_xy.append(cy)
            dy_xy.append(dy)

        # Get maximum Cy of both directions
        # Only cy is consistent when designing both directions, the rest of the parameters are unique to the direction
        # cy = geo_mean(cy_xy)
        cy = max(cy_xy)

        return cy, dy_xy, spo2ida_data

    def run_analysis(self, solution, forces, direction, hinge):
        """
        Runs analysis via OpenSees in 3D
        :param solution: dict                       Building cross-section information
        :param forces: dict                         Acting lateral forces in 1 direction
        :param direction: int                       Direction to run analysis in
        :param hinge: dict                          Hinge models for the entire building
        :return: dict                               Demands on structural components in 1 direction
        """
        # Get the system configuration
        system = self.ipbsd.data.configuration.lower()

        # Muto's approach is not implemented herein (it was not a recommended option anyway
        # Current: Only ELFM and ELFM+gravity are supported
        if self.analysis_type == 2:
            demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), fstiff=self.fstiff,
                                              hinge=hinge, direction=direction, system=system)
        elif self.analysis_type == 3:
            demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                              fstiff=self.fstiff, hinge=hinge, direction=direction, system=system)
        else:
            raise ValueError("[EXCEPTION] Incorrect analysis type...")
        return demands

    def get_critical_designs(self, hinge_models_x, hinge_models_y):
        """
        Modify hinge elements of external seismic columns to the strongest (larger My) from designs of both directions
        :param hinge_models_x:
        :param hinge_models_y:
        :return:
        """
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

        return hinge_models_x, hinge_models_y

    def design_building(self, cy, dy, solution, modes, table_sls, gravity_demands, hinge=None):
        """
        Design the structural components of the building
        :param cy: float                        Yield strength to design the building for
        :param dy: list                         Yield displacements in both directions
        :param solution: dict                   Structural component cross-sections
        :param modes: dict                      Modal properties in both directions
        :param table_sls: dict                  DBD tables
        :param gravity_demands: dict            Placeholder for gravity demands
        :param hinge: dict                      Hinge models
        :return: dict
        """
        # Initialize demands
        demands = {"x": None, "y": None}
        # Initialize details
        details = {"x": None, "y": None}
        # Initialize hinge models
        if hinge is None:
            hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

        # Get the acting lateral forces based on cy for each direction
        forces = {"x": self.ipbsd.get_action(solution["x_seismic"], cy, pd.DataFrame.from_dict(table_sls["x"]),
                                             self.gravity_loads, self.analysis_type),
                  "y": self.ipbsd.get_action(solution["y_seismic"], cy, pd.DataFrame.from_dict(table_sls["y"]),
                                             self.gravity_loads, self.analysis_type)}

        # Demands on all elements of the system where plastic hinge information is missing
        # (or is related to the previous iteration)
        for key in demands.keys():
            d = 0 if key == "x" else 1
            demands[key] = self.run_analysis(solution, forces[key], direction=d, hinge=hinge)

            # Update the Gravity demands
            gravity_demands[key] = demands[key]["gravity"]

            # Design the structural elements
            designs, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
                self.ipbsd.design_elements(demands[key][key + "_seismic"], solution[key + "_seismic"], modes, dy[d],
                                           cover=self.rebar_cover, direction=d, est_ductilities=False)
            details[key] = {"details": designs, "hinge_models": hinge_models, "mu_c": mu_c, "mu_f": mu_f,
                            "warnMax": warnMax, "warnMin": warnMin, "warnings": warnings, "cy": cy, "dy": dy}

        # Design the central elements (envelope of both directions)
        hinge_gravity, warn_gravity = self.ipbsd.design_elements(gravity_demands, solution["gravity"], None, None,
                                                                 cover=self.rebar_cover, direction=0, gravity=True,
                                                                 est_ductilities=False)

        # Take the critical of hinge models from both directions
        hinge_x, hinge_y = self.get_critical_designs(details["x"]["hinge_models"], details["y"]["hinge_models"])

        # """Rerun demand estimation and designs using the identified hinge models"""
        # # hinge models
        # hinge = {"x_seismic": hinge_x, "y_seismic": hinge_y, "gravity": hinge_gravity}
        # for key in details.keys():
        #     d = 0 if key == "x" else 1
        #     demands[key] = self.run_analysis(solution, forces[key], direction=d, hinge=hinge)
        #
        #     # Update the Gravity demands
        #     gravity_demands[key] = demands[key]["gravity"]
        #
        #     # Design the structural elements
        #     designs, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
        #         self.ipbsd.design_elements(demands[key][key + "_seismic"], solution[key + "_seismic"], modes, dy[d],
        #                                    cover=self.rebar_cover, direction=d, est_ductilities=False)
        #     details[key] = {"details": designs, "hinge_models": hinge_models, "mu_c": mu_c, "mu_f": mu_f,
        #                     "warnMax": warnMax, "warnMin": warnMin, "warnings": warnings, "cy": cy, "dy": dy}
        #
        # # Design the central elements (envelope of both directions)
        # hinge_gravity, warn_gravity = self.ipbsd.design_elements(gravity_demands, solution["gravity"], None, None,
        #                                                          cover=self.rebar_cover, direction=0, gravity=True,
        #                                                          est_ductilities=False)
        #
        # # Take the critical of hinge models from both directions
        # hinge_x, hinge_y = self.get_critical_designs(details["x"]["hinge_models"], details["y"]["hinge_models"])

        details["gravity"] = {"hinge_models": hinge_gravity, "warnings": warn_gravity}

        # Update the existing hinge models
        details["x"]["hinge_models"] = hinge_x
        details["y"]["hinge_models"] = hinge_y
        return details

    def generate_initial_solutions(self, opt_sol, opt_modes, omega, table_sls):
        """
        Master file to run for each direction/iteration (only for 3D modelling)
        :param opt_sol: dict                        Dictionary containing structural element cross-section information
        :param opt_modes: dict                      Modal properties of the solution
        :param omega: float                         Overstrength factor
        :param table_sls: DataFrame                 DBD table at SLS
        :return: Design outputs and Demands on gravity (internal elements)
        """
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

        # Initialize period to use (reset when running for Y direction)
        self.period_to_use = None

        # Generate initial solutions for both directions
        # For each primary direction of building, apply lateral loads and design the structural elements
        print("[PHASE] Target for MAFC...")
        cy, dy, spo2ida_data = self.target_for_mafc(opt_sol, omega)

        """Get action and demands"""
        print("[PHASE] Design structural components...")
        self.details = self.design_building(cy, dy, opt_sol, opt_modes, table_sls, demands_gravity)

        return self.details

    def run_ma(self, solution, hinge, period_limits, tol=1.05, spo_period=None, do_corrections=True):
        """
        Creates a nonlinear model and runs Modal Analysis with the aim of correcting solution and fundamental period
        :param solution: dict                   Cross-sections
        :param hinge: dict                      Hinge models
        :param period_limits: dict              Period limits from IPBSD
        :param tol: float                       Tolerance for period
        :param spo_period: float                Period identified via SPO analysis
        :param do_corrections: bool             Whether corrections are necessary if period conditions is not met
        :return:
        """
        print("[MA] Running modal analysis")
        # 1st index refers to X, and 2nd index refers to Y
        model_periods, modalShape, gamma, mstar = self.ipbsd.ma_analysis(solution, hinge, None, self.fstiff)

        # If SPO periods are identified
        if spo_period is not None:
            model_periods = spo_period

        # Only the upper period is checked, as it is highly unlikely to get a period lower than the lower limit
        # Raise a warning for now, for a developer -> add a solution (this is a bit of a complicated issue, as the
        # lowering of cross-section dimensions may have negative effects on other design aspects). So we will deal with
        # it when we cross that river

        # Upper period limits in each direction
        upper_limits = [period_limits["x"][1], period_limits["y"][1]]
        '''
        Currently a new solution is being sought increasing c-s dimensions by 0.5 - a rather brute force approach.
        However, even brute force is not expensive.
        Temporary reductions if within certain limits
        Elements allowed for modifications:
        1. Internal columns of external frames
        2. Beams heights
        '''
        # Check along X direction
        if model_periods[0] > tol * upper_limits[0] and do_corrections:
            for st in range(self.ipbsd.data.nst):
                # Internal columns of external frames along X
                solution["x_seismic"][f"hi{st + 1}"] += 0.05
                # Gravity beams along X
                solution["gravity"][f"hx{st + 1}"] += 0.05
                # Gravity columns
                solution["gravity"][f"hi{st + 1}"] += 0.05
                # If within XX% do not increase beam heights along corner frames
                if tol * upper_limits[0] / model_periods[0] < 0.75:
                    solution["x_seismic"][f"h{st + 1}"] += 0.05
                # If any of the cross-section dimensions is beyond an undesirable value, raise a warning
                vlist = [solution["x_seismic"][f"hi{st + 1}"], solution["x_seismic"][f"h{st + 1}"],
                         solution["gravity"][f"hx{st + 1}"]]
                if not all(v < 0.95 for v in vlist):
                    print("[WARNING] Cross-section dimensions are above 0.9m.")

            # Raise a warning
            warnT_x = True
        else:
            warnT_x = False

        # Check along Y direction
        if model_periods[1] > tol * upper_limits[1] and do_corrections:
            for st in range(self.ipbsd.data.nst):
                # Internal columns of external frames along Y
                solution["y_seismic"][f"hi{st + 1}"] += 0.05
                # Gravity beams along Y
                solution["gravity"][f"hy{st + 1}"] += 0.05
                # Only if no warning was raised along Y (as we don't want to increase the sections twice)
                if model_periods[0] <= tol * upper_limits[0]:
                    # Gravity columns
                    solution["gravity"][f"hi{st + 1}"] += 0.05
                # If within XX% do not increase beam heights along corner frames
                if tol * upper_limits[1] / model_periods[1] < 0.75:
                    solution["y_seismic"][f"h{st + 1}"] += 0.05
                # If any of the cross-section dimensions is beyond an undesirable value, raise a warning
                vlist = [solution["y_seismic"][f"hi{st + 1}"], solution["y_seismic"][f"h{st + 1}"],
                         solution["gravity"][f"hy{st + 1}"]]
                if not all(v < 0.95 for v in vlist):
                    print("[WARNING] Cross-section dimensions are above 0.9m.")

            # Raise a warning
            warnT_y = True
        else:
            warnT_y = False
        # Period warnings in any direction?
        self.warnT = warnT_x or warnT_y

        # If warning was raised, perform modal analysis to recalculate the modal parameters
        # Note: Even though the period might not yet be within the limits, it is recommended to carry on as SPO analysis
        # will provide the actual secant to yield periods
        if self.warnT:
            model_periods, modalShape, gamma, mstar = self.ipbsd.ma_analysis(solution, hinge, None, self.fstiff)

        # Update the modal parameters in solution
        solution["x_seismic"]["T"] = model_periods[0]
        solution["x_seismic"]["Part Factor"] = gamma[0]
        solution["x_seismic"]["Mstar"] = mstar[0]

        solution["y_seismic"]["T"] = model_periods[1]
        solution["y_seismic"]["Part Factor"] = gamma[1]
        solution["y_seismic"]["Mstar"] = mstar[1]

        return model_periods, modalShape, gamma, mstar, solution

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
        if direction == "gravity":
            opt = opt_sol[direction]
        else:
            opt = opt_sol[direction + "_seismic"]

        # Create an empty dictionary of warnings
        w = {x: False for x in opt.keys()}

        # After modifications, equally between c-s of two storeys might not hold
        # Increment for cross-section modifications for elements with warnings
        increment = 0.05
        if direction != "gravity":
            # Columns
            for ele in warnings["MAX"]["Columns"]:
                if warnings["MAX"]["Columns"][ele] == 1:
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
                        # Increase section cross-section
                        opt[f"b{storey}"] = opt[f"he{storey}"]
                        opt[f"h{storey}"] = opt[f"h{storey}"] + increment
                        w[f"h{storey}"] = True

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

            # Update the optimal solution
            opt_sol[direction + "_seismic"] = opt
            opt_sol[perp_dir + "_seismic"] = opt_perp
            opt_sol["gravity"] = opt_gr
        else:
            # Only central components are affected
            # Columns
            for ele in warnings["MAX"]["Columns"]:
                if warnings["MAX"]["Columns"][ele] == 1:
                    # A warning was found at storey level
                    storey = ele[-1]
                    # Modify the cross-section dimensions
                    opt[f"hi{storey}"] = opt[f"hi{storey}"] + increment
                    w[f"hi{storey}"] = True

            # Beams
            for i in warnings["MAX"]["Beams"]:
                for ele in warnings["MAX"]["Beams"][i]:
                    storey = ele[1]
                    d = ele[-1]
                    if warnings["MAX"]["Beams"][i][ele] == 1:
                        # Modify the cross-section dimensions
                        opt[f"h{d}{storey}"] = opt[f"h{d}{storey}"] + increment
                        w[f"h{d}{storey}"] = True
            # Update the optimal solution
            opt_sol["gravity"] = opt

        return opt_sol

    def conservative_spo_shape(self, spo, residual=0.25):
        x = spo[0]
        y = spo[1]

        # All negatives to zero
        y[y < 0] = 0.0

        # Get maximum point for reference
        Vmax = max(y)

        # Get initial stiffness
        m1 = 0.2 * Vmax
        d1 = x[getIndex(m1, y)]
        stiff_elastic = m1 / d1

        # Get the yield point
        slopes = y / x
        stfIdx = np.where(slopes[1:] < 0.85 * stiff_elastic)[0][0]
        xint = x[stfIdx + 1]
        yint = y[stfIdx + 1]

        # Get the point of softening
        for i in range(len(x) - 1):
            stf = (y[i + 1] - y[i]) / (x[i] - x[i + 1])

            if stf > 50000:
                # High spikes
                ymax = y[i]
                xmax = x[i]
                break

        if "ymax" not in locals():
            ymax = max(y)
            xmax = x[getIndex(ymax, y)]

        # Make sure yield point is not larger than max point
        if yint > ymax:
            yint = ymax

        # # Residual point
        # yres = max(y[-1], residual * yint)
        # xres = x[i + getIndex(-yres, -y[i + 1:])]
        #
        # for i in range(len(x) - 1, 0, -1):
        #     if y[i] <= 0.0:
        #         y[i] = residual * yint
        #     if y[i - 1] / y[i] > 1.2 and y[i-1] > residual * yint:
        #         xres = x[i - 1]
        #         yres = y[i - 1]
        #         break
        # try:
        #     if yres > 0.35 * yint:
        #         yres = 0.35 * yint
        #         xres = x[i + getIndex(-yres, -y[i + 1:])]
        # except:
        #     pass

        # # Now, identify the residual strength point (here defined at V=0)
        # yres = max(y[-1], yint * residual)
        # idx = getIndex(1.01 * yres, y[::-1])
        # xres = x[::-1][idx]
        # # Getting the actual residual strength and corresponding displacement
        # ymin = yres
        #
        # # Select the softening slope until residual displacement
        # # Fitting based on the area under the softening slope
        # y_soft = y[getIndex(Vmax, y): getIndex(xres, x)]
        # nbins = len(y_soft) - 1
        # dx = (xres - xmax) / nbins
        # area_soft = np.trapz(y_soft, dx=dx)
        # xmin = 2 * area_soft / (Vmax + ymin) + xmax
        #
        # xres = xmin
        # yres = ymin

        # Using the Stiffness up till 0.8*ymax
        y_80 = 0.6 * yint
        idx = getIndex(1.01 * y_80, y[::-1])
        x_80 = x[::-1][idx]

        yres = residual * yint
        xres = (ymax - yres) * (x_80 - xmax) / (ymax - y_80) + xmax

        # Define the curve
        d = np.array([0., xint, xmax, xres])
        v = np.array([0., yint, ymax, yres])

        return d, v

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
        ymin = yres

        # ymin = y[-1]
        #
        # # Avoid negative residual strength and zero as residual strength
        # cnt = 2
        # while ymin <= 0:
        #     ymin = y[-cnt]
        #     cnt += 1

        # xmin = (Vmax - ymin) * (xres - dmax) / (Vmax - yres) + dmax

        # Select the softening slope until residual displacement
        # Fitting based on the area under the softening slope
        y_soft = y[getIndex(Vmax, y): getIndex(xres, x)]
        nbins = len(y_soft) - 1
        dx = (xres - dmax) / nbins
        area_soft = np.trapz(y_soft, dx=dx)
        xmin = 2 * area_soft / (Vmax + ymin) + dmax

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

    def run_spo(self, solution, hinge, vy, pattern, omega, direction="x"):
        """
        Create a nonlinear model in OpenSees and runs SPO
        :param solution: DataFrame              Design solution
        :param hinge: DataFrame                 Nonlinear hinge models
        :param vy: float                        Design base shear of the MDOF system (excludes Omega)
        :param pattern: list                    First-mode shape from MA
        :param omega: float                     Overstrength factor
        :param direction: str                   Direction of pushover action
        :return: tuple                          SPO outputs (Top displacement vs. Base Shear)
        :return: tuple                          Idealized SPO curve fit (Top displacement vs. Base Shear)
        :return: float                          Overstrength factor
        """
        d = 0 if direction == "x" else 1
        spo_results = self.ipbsd.spo_opensees(solution, hinge, None, self.fstiff, pattern, direction=d)

        # Get the idealized version of the SPO curve and create a warningSPO = True if the assumed shape was incorrect
        # DEVELOPER TOOL
        new_fitting = True
        if new_fitting:
            d, v = self.conservative_spo_shape(spo_results)
        else:
            d, v = self.derive_spo_shape(spo_results, residual=0.3)

        # Actual overstrength
        omega_new = v[1] / vy

        # Verify that overstrength is correct
        # Additionally, if a threshold of warnings of WarnMin is exceeded, do not update the overstrength
        # As demands are too low for the given cross-section dimensions and the overstrength value will just snowball
        # without impact
        # (the only impact of it being the meaningless decrease of cy for the same level of increase in overstrength)
        # Subject to change and calibration (currently Min check only on columns)
        warnings_min = sum(self.details["x"]["warnings"]["MIN"]["Columns"].values()) + \
                       sum(self.details["y"]["warnings"]["MIN"]["Columns"].values()) + \
                       sum(self.details["gravity"]["warnings"]["warnings"]["MIN"]["Columns"].values())
        n_elements = len(self.details["x"]["warnings"]["MIN"]["Columns"]) + \
                     len(self.details["y"]["warnings"]["MIN"]["Columns"]) + \
                     len(self.details["gravity"]["warnings"]["warnings"]["MIN"]["Columns"])

        if warnings_min / n_elements < 0.5:
            if omega * 0.95 <= omega_new <= omega * 1.05:
                omega_new = omega
                self.omegaWarn = False
            else:
                self.omegaWarn = True
        else:
            omega_new = omega
            self.omegaWarn = False

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
        self.spo_validate = self.compare_areas(self.spo_shape[direction], spo_data_updated, tol=0.2)

        # Update the SPO parameters
        if not self.spo_validate:
            self.spo_shape[direction] = spo_data_updated
        else:
            self.spo_shape[direction] = self.spo_shape[direction]

        return spo_results, [d, v], omega_new

    def run_iterations(self, opt_sol, opt_modes, period_limits, table, maxiter=10, omega=1.0):
        """
        Runs iterations in order to find a suitable solution (uses both directions at once, instead of coming up with a
        solution sequentially)
        I. Before Iterations
        1. Run Modal analysis based on initial solutions before starting the iterations
        Make sure that model periods are within the period limits
        II. Start the iterations
        2. Corrections because of unsatisfactory detailing (warnMax)
        3. Run SPO analysis in both directions, identify SPO Period and SPO shape
        If SPO Period not met raise warnT, if shape not satisfactory raise self.spo_validate
        III. Start Second Iteration
        4. Run Modal Analysis and checking for Period using SPO period
        Raise WarnT if period not satisfactory
        5. Target for MAFC
        6. Design the building
        7. Corrections because of unsatisfactory detailing (warnMax)
        8. Run SPO analysis in both directions, identify SPO Period and SPO shape
        :return:
        """
        # Initial assumptions for modal parameters
        mstar = np.array([opt_sol["x_seismic"]["Mstar"], opt_sol["y_seismic"]["Mstar"]])
        gamma = np.array([opt_sol["x_seismic"]["Part Factor"], opt_sol["y_seismic"]["Part Factor"]])
        modes = opt_modes["Modes"]

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

        if isinstance(omega, float):
            omega = [omega] * 2

        # Retrieve necessary information from detailing results
        hinge_gr = self.details["gravity"]["hinge_models"]
        hinge_x = self.details["x"]["hinge_models"]
        hinge_y = self.details["y"]["hinge_models"]
        hinge_models = {"x_seismic": hinge_x, "y_seismic": hinge_y, "gravity": hinge_gr}
        cy = self.details["x"]["cy"]

        # # Run modal analysis
        # print("[PHASE] Running Modal Analysis...")
        # # Optimal solution might change here, however hinge element cross-sections are not updated yet
        # periods, modes, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, period_limits)

        # Iterate until all conditions are met
        cnt = 0
        # Initialize SPO period
        spo_period = None
        # Reading from file the SPO parameters (initialize)
        read = True

        # Any warnings because of detailing?
        warnMax = self.details["x"]["warnMax"] or self.details["y"]["warnMax"] or \
                  self.details["gravity"]["warnings"]["warnMax"]

        while (self.warnT or warnMax or not self.spo_validate or self.omegaWarn) and cnt + 1 <= maxiter:

            # Iterations skipping first iteration
            if (not self.spo_validate or self.omegaWarn or self.warnT) and cnt > 0:
                # Rerun Modal analysis if warnT was raised
                print("[PHASE] Running Modal Analysis...")
                # Optimal solution might change, but the hinge_models cross-sections are not changed yet
                # It will be modified accordingly during design of the building
                periods, modes, gamma, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, period_limits,
                                                                    spo_period=spo_period)

                # Target for MAFC
                print("[PHASE] Target MAFC in both directions...")
                cy, dy, spo2ida_data = self.target_for_mafc(opt_sol, omega, read=read)

                # Design the building
                print("[PHASE] Design and detail the building...")
                self.details = self.design_building(cy, dy, opt_sol, modes, table, demands_gravity)

                # Retrieve necessary information from detailing results
                hinge_gr = self.details["gravity"]["hinge_models"]
                hinge_x = self.details["x"]["hinge_models"]
                hinge_y = self.details["y"]["hinge_models"]
                hinge_models = {"x_seismic": hinge_x, "y_seismic": hinge_y, "gravity": hinge_gr}

                # Any warnings because of detailing?
                warnMax = self.details["x"]["warnMax"] or self.details["y"]["warnMax"] or \
                          self.details["gravity"]["warnings"]["warnMax"]

                # Correction if unsatisfactory detailing: modifying only towards increasing c-s
                # Direction X
                if self.details["x"]["warnMax"]:
                    warnings = self.details["x"]["warnings"]
                    opt_sol = self.seek_solution(warnings, opt_sol, direction="x")
                # Direction Y
                if self.details["y"]["warnMax"]:
                    warnings = self.details["y"]["warnings"]
                    opt_sol = self.seek_solution(warnings, opt_sol, direction="y")
                # Central components
                if self.details["gravity"]["warnings"]["warnMax"]:
                    warnings = self.details["gravity"]["warnings"]["warnings"]
                    opt_sol = self.seek_solution(warnings, opt_sol, direction="gravity")

                if warnMax:
                    # If the optimal solutions were modified new design of the building should be carried out
                    self.details = self.design_building(cy, dy, opt_sol, modes, table, demands_gravity)
                    # Retrieve necessary information from detailing results
                    hinge_gr = self.details["gravity"]["hinge_models"]
                    hinge_x = self.details["x"]["hinge_models"]
                    hinge_y = self.details["y"]["hinge_models"]
                    hinge_models = {"x_seismic": hinge_x, "y_seismic": hinge_y, "gravity": hinge_gr}

                    # Any warnings because of detailing?
                    warnMax = self.details["x"]["warnMax"] or self.details["y"]["warnMax"] or \
                              self.details["gravity"]["warnings"]["warnMax"]
                    print("[RERUN COMPLETE] New design solution was selected due to unsatisfactory detailing...")

            """Create an OpenSees model and run static pushover - iterate analysis.
            Acts as the first estimation of secant to yield period. Second step after initial modal analysis.
            Here, the secant to yield period might vary from initial modal period. 
            Therefore, we need to use the former one."""
            # For a single frame assumed yield base shear
            vy_design = cy * gamma * mstar * 9.81
            omega_cache = omega.copy()
            spo_shape_cache = self.spo_shape
            spo_pattern = np.round(modes, 2)
            # Run SPO analysis
            print("[SPO] Starting SPO analysis...")
            spo_x, idealized_x, omega[0] = self.run_spo(opt_sol, hinge_models, vy_design[0], spo_pattern[:, 0],
                                                        omega[0], direction="x")
            spo_y, idealized_y, omega[1] = self.run_spo(opt_sol, hinge_models, vy_design[1], spo_pattern[:, 1],
                                                        omega[1], direction="y")
            for i in range(len(omega)):
                if omega[i] < 1.0:
                    omega[i] = 1.0

            # Calculate the period as secant to yield period
            spo_period = np.zeros((2, ))
            spo_period[0] = 2 * np.pi * np.sqrt(mstar[0] / (idealized_x[1][1] / idealized_x[0][1]))
            spo_period[1] = 2 * np.pi * np.sqrt(mstar[1] / (idealized_y[1][1] / idealized_y[0][1]))
            self.period_to_use = spo_period

            # Record OpenSees outputs
            self.model_outputs = {"MA": {"T": spo_period, "modes": spo_pattern, "gamma": gamma, "mstar": mstar},
                                  "SPO": {"x": spo_x, "y": spo_y},
                                  "SPO_idealized": {"x": idealized_x, "y": idealized_y}}

            if not spo_period[0] <= period_limits["x"][1] * 1.05:
                # Even though SPO period might not match modal period, the condition is still satisfying
                warnT_x = True
            else:
                # If SPO based secant to yield period is not within the tolerance of upper period limit
                warnT_x = False

            if not spo_period[1] <= period_limits["y"][1] * 1.05:
                # Even though SPO period might not match modal period, the condition is still satisfying
                warnT_y = True
            else:
                # If SPO based secant to yield period is not within the tolerance of upper period limit
                warnT_y = False
            self.warnT = warnT_x or warnT_y

            # Reading SPO parameters from file (Set to False, as the actual shape is already identified)
            read = False

            print("[SUCCESS] Static pushover analysis was successfully performed.")

            print(f"[ITERATION {cnt + 1} END]")

            # Increase count of iterations
            cnt += 1

        if cnt == maxiter:
            print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": gamma, "Mstar": mstar, "Period range": period_limits,
                         "overstrength": omega, "cy": cy, "dy": dy, "spo2ida": self.spo_shape}

        return ipbsd_outputs, spo2ida_data, opt_sol, modes, self.details, hinge_models, self.model_outputs
