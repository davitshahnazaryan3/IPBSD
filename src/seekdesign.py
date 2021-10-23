from scipy.optimize import fsolve
import pandas as pd

from analysis.detailing import Detailing
from src.MAFC import MAFC
from src.crossSectionSpace import CrossSectionSpace
from tools.spo2ida import SPO2IDA
from analysis.action import Action
from analysis.openseesrun import OpenSeesRun
from analysis.analysisMethods import run_opensees_analysis
from utils.ipbsd_utils import compare_areas
from utils.seek_design_utils import *
from utils.spo2ida_utils import read_spo_data


class SeekDesign:
    """For 3D modelling only"""

    def __init__(self, spo_filename, target_mafc, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 gravity_loads, system, data, hazard, export_directory):
        """
        Initialize iterations
        :param spo_filename: str                        Path to .csv containing SPO shape assumptions
        :param target_mafc: float                       Target MAFC
        :param analysis_type: int                       Type of elastic analysis for design purpose
        :param damping: float                           Damping ratio
        :param num_modes: int                           Number of modes for RMSA
        :param fstiff: float                            Stiffness reduction factor unless hinge information is available
        :param rebar_cover: float                       Reinforcement cover
        :param gravity_loads: dict                      Gravity loads to be applied
        :param system: str                              Perimeter or Space
        :param data: dict                               Input arguments of IPBSD
        :param hazard: dict                             Hazard curves
        :param export_directory: str                    Path to export outputs
        """
        self.spo_filename = spo_filename
        self.target_mafc = target_mafc
        self.analysis_type = analysis_type
        self.damping = damping
        self.num_modes = num_modes
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.gravity_loads = gravity_loads
        self.system = system
        self.data = data
        self.hazard = hazard
        self.export_directory = export_directory

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
        # Period to be used when performing important tools (list)
        self.period_to_use = None
        # Initialize detailing results
        self.details = None
        # print stuff
        self.pflag = False
        # 3D modelling
        self.flag3d = True

    def run_elastic_analysis(self, solution, forces, hinge, direction):
        """
        Runs analysis via OpenSees in 3D
        :param solution: dict                       Building cross-section information
        :param forces: dict                         Acting lateral forces in 1 direction
        :param hinge: dict                          Hinge models for the entire building
        :param direction: int                       Direction of action, 0=x, 1=y
        :return: dict                               Demands on structural components in 1 direction
        """
        # Current: Only ELFM and ELFM+gravity are supported - more methods to be added
        # call the OpenSees model
        op = OpenSeesRun(self.data, solution, fstiff=self.fstiff, hinge=hinge, direction=direction,
                         system=self.system, pflag=self.pflag)

        if self.analysis_type == 2:
            # no gravity loads
            demands = op.run_elastic_analysis(self.analysis_type, lateral_action=list(forces["Fi"]))

        elif self.analysis_type == 3:
            # gravity loads included
            demands = op.run_elastic_analysis(self.analysis_type, lateral_action=list(forces["Fi"]),
                                              grav_loads=list(forces["G"]))

        else:
            raise ValueError("[EXCEPTION] Incorrect analysis type...")

        return demands

    def run_ma(self, solution, hinge, period_limits, direction, tol=1.05, spo_period=None, do_corrections=True):
        """
        Creates a nonlinear model and runs Modal Analysis with the aim of correcting solution and fundamental period
        :param solution: dict                   Cross-sections
        :param hinge: dict                      Hinge models
        :param period_limits: dict              Period limits from IPBSD
        :param direction: int                   Direction of action
        :param tol: float                       Tolerance for period
        :param spo_period: float                Period identified via SPO analysis
        :param do_corrections: bool             Whether corrections are necessary if period conditions is not met
        :return:
        """
        print("[MA] Running modal analysis")
        # 1st index refers to X, and 2nd index refers to Y
        model_periods, modalShape, part_factor, mstar = run_opensees_analysis(direction, solution, hinge, self.data,
                                                                              None, self.fstiff, self.flag3d)

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
        1. Internal columns of analysis frames
        2. Beams heights
        '''
        # Check along X direction
        if model_periods[0] > tol * upper_limits[0] and do_corrections:
            solution = find_solution(self.data.nst, model_periods[0], solution, upper_limits[0], tol, "x")

            # Raise a warning
            warnT_x = True
        else:
            # Raise no warning
            warnT_x = False

        # Check along Y direction
        if model_periods[1] > tol * upper_limits[1] and do_corrections:
            solution = find_solution(self.data.nst, model_periods[1], solution, upper_limits[1], tol, "y",
                                     period_1=model_periods[0], limit_1=upper_limits[0])

            # Raise a warning
            warnT_y = True
        else:
            # Raise no warning
            warnT_y = False

        # Period warnings in any direction?
        self.warnT = warnT_x or warnT_y

        # If warning was raised, perform modal analysis to recalculate the modal parameters
        # Note: Even though the period might not yet be within the limits, it is recommended to carry on as SPO analysis
        # will provide the actual secant to yield periods
        if self.warnT:
            model_periods, modalShape, part_factor, mstar = run_opensees_analysis(direction, solution, hinge, self.data,
                                                                                  None, self.fstiff, self.flag3d)

        # Update the modal parameters in solution
        solution["x_seismic"]["T"] = model_periods[0]
        solution["x_seismic"]["Part Factor"] = part_factor[0]
        solution["x_seismic"]["Mstar"] = mstar[0]

        solution["y_seismic"]["T"] = model_periods[1]
        solution["y_seismic"]["Part Factor"] = part_factor[1]
        solution["y_seismic"]["Mstar"] = mstar[1]

        return model_periods, modalShape, part_factor, mstar, solution

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
        spo_results = run_opensees_analysis(d, solution, hinge, self.data, None, self.fstiff, self.flag3d, pattern)

        # Get the idealized version of the SPO curve and create a warningSPO = True if the assumed shape was incorrect
        # DEVELOPER TOOL
        new_fitting = True
        if new_fitting:
            d, v = get_conservative_spo_shape(spo_results)
        else:
            d, v = derive_spo_shape(spo_results, residual=0.3)

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
        spo_data_updated = get_spo2ida_parameters(d, v, self.period_to_use)

        # Check whether the parameters vary from the original assumption (if True, then tolerance met)
        # NOTE: spo shape might loop around two scenarios (need a better technique to look for a solution)
        # It is primarily due to some low values (e.g. muC), where the large variations of e.g. a do not have a
        # significant impact on the final results, however are hard to fit
        # self.spo_validate = all(list(map(self.compare_value, self.spo_data.values(), spo_data_updated.values())))

        # Alternatively compare the area under the normalized SPO curves (possible update: compare actual SPO curves)
        self.spo_validate = compare_areas(self.spo_shape[direction], spo_data_updated, tol=0.2)

        # Update the SPO parameters
        if not self.spo_validate:
            self.spo_shape[direction] = spo_data_updated
        else:
            self.spo_shape[direction] = self.spo_shape[direction]

        return spo_results, [d, v], omega_new

    @staticmethod
    def run_spo2ida(data):
        mc, a, ac, r, mf, period, pw = map(data.get, ('mc', 'a', 'ac', 'r', 'mf', 'T', 'pw'))

        model = SPO2IDA(mc, a, ac, r, mf, period, pw)
        R16, R50, R84, idacm, idacr, spom, spor = model.run_spo2ida_allT()
        output = {'R16': R16, 'R50': R50, 'R84': R84, 'idacm': idacm, 'idacr': idacr, 'spom': spom, 'spor': spor}
        return output

    def verify_mafc(self, period, spo2ida, part_factor, omega):
        """
        optimizes for a target mafc
        :param period: float                        Fundamental period of the structure
        :param spo2ida: dict                        Dictionary containing SPO2IDA results
        :param part_factor: float                   First mode participation factor
        :param omega: float                         Overstrength factor
        :return: float, float                       Spectral acceleration [g] and displacement [m] at yield
        """
        r = [spo2ida['R16'], spo2ida['R50'], spo2ida['R84']]
        Hs, sa_hazard = self.hazard[2][int(round(period * 10))], self.hazard[1][int(round(period * 10))]

        m = MAFC(r, self.target_mafc, part_factor, Hs, sa_hazard, omega, True)
        fsolve(m.objective, x0=np.array([0.02]), factor=0.1)

        # Yield displacement for ESDOF
        dy = float(m.cy) * 9.81 * (period / 2 / np.pi) ** 2
        cy = float(m.cy)

        return cy, dy

    def target_for_mafc(self, solution, overstrength, read=True):
        """
        Look for a spectral acceleration at yield to target for MAFC
        :param solution: dict               Cross-section information of the building's structural components
        :param overstrength: float          Overstrength factor
        :param read: bool                   Read from file?
        :return cy: float                   Maximum Yield spectral acceleration in g (max of both directions)
        :return dy_xy: list                 Yield displacements in both directions
        :return spo2ida_data: dict          SPO2IDA results
        """
        # Overstrength
        if not isinstance(overstrength, list):
            overstrength = [overstrength] * 2
        else:
            overstrength = overstrength

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
            self.spo_shape["x"] = read_spo_data(self.spo_filename)
            self.spo_shape["y"] = read_spo_data(self.spo_filename)

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
            spo2ida_data[key] = self.run_spo2ida(self.spo_shape[key])
            cy, dy = self.verify_mafc(period[i], spo2ida_data[key], part_factor[i], overstrength[i])
            cy_xy.append(cy)
            dy_xy.append(dy)

        # Get maximum Cy of both directions
        # Only cy is consistent when designing both directions, the rest of the parameters are unique to the direction
        # cy = geo_mean(cy_xy)
        # DEVELOPER TOOL
        same_cy = True
        if same_cy:
            cy = max(cy_xy)
        else:
            cy = np.array(cy_xy)

        return cy, dy_xy, spo2ida_data

    def get_acting_loads(self, solution, table, cyx, cyy, num_modes=None, opt_modes=None, modal_sa=None):
        a = Action(self.data, self.analysis_type, self.gravity_loads, num_modes=num_modes, opt_modes=opt_modes,
                   modal_sa=modal_sa)

        forces = {
            "x": a.forces(solution["x_seismic"], pd.DataFrame.from_dict(table["x"]), cyx),
            "y": a.forces(solution["y_seismic"], pd.DataFrame.from_dict(table["y"]), cyy)
        }
        return forces

    def design_building(self, cy, dy, solution, modes, table_sls, gravity_demands, hinge=None):
        """
        Design the structural components of the building
        :param cy: float                        Yield strength to design the building for
        :param dy: list                         Yield displacements in both directions
        :param solution: dict                   Structural component cross-sections
        :param modes: dict                      Modal properties in both directions
        :param table_sls: dict                  DBD tables at serviceability limit state, SLS
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
        if not isinstance(cy, float):
            cyx = cy[0]
            cyy = cy[1]
        else:
            cyx = cyy = cy

        # Get acting loads
        forces = self.get_acting_loads(solution, table_sls, cyx, cyy)

        # Demands on all elements of the system where plastic hinge information is missing
        # (or is related to the previous iteration)
        for key in demands.keys():
            d = 0 if key == "x" else 1
            # Run analysis in each direction sequentially
            demands[key] = self.run_elastic_analysis(solution, forces[key], hinge, d)

            # Update the Gravity demands
            gravity_demands[key] = demands[key]["gravity"]

            # Design the structural elements
            designs, hinge_models, mu_c, mu_f, warnMax, warnMin, warnings = \
                self.design_elements(demands[key][key + "_seismic"], solution[key + "_seismic"], modes, dy[d],
                                     cover=self.rebar_cover, direction=d, est_ductilities=False)
            details[key] = {"details": designs, "hinge_models": hinge_models, "mu_c": mu_c, "mu_f": mu_f,
                            "warnMax": warnMax, "warnMin": warnMin, "warnings": warnings, "cy": cy, "dy": dy}

        # Design the central elements (envelope of both directions)
        hinge_gravity, warn_gravity = self.design_elements(gravity_demands, solution["gravity"], None, None,
                                                           cover=self.rebar_cover, direction=0, gravity=True,
                                                           est_ductilities=False)

        # Take the critical of hinge models from both directions
        hinge_x, hinge_y = get_critical_designs(details["x"]["hinge_models"], details["y"]["hinge_models"])

        details["gravity"] = {"hinge_models": hinge_gravity, "warnings": warn_gravity}

        # Update the existing hinge models
        details["x"]["hinge_models"] = hinge_x
        details["y"]["hinge_models"] = hinge_y
        return details

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

    def initialize_demands(self):
        bx_gravity = np.zeros((self.data.nst, self.data.n_bays, len(self.data.spans_y) - 1))
        by_gravity = np.zeros((self.data.nst, self.data.n_bays - 1, len(self.data.spans_y)))
        c_gravity = np.zeros((self.data.nst, self.data.n_bays - 1, len(self.data.spans_y) - 1))
        gravity_temp = {"Beams_x": {"M": {"Pos": bx_gravity.copy(), "Neg": bx_gravity.copy()},
                                    "N": bx_gravity.copy(), "V": bx_gravity.copy()},
                        "Beams_y": {"M": {"Pos": by_gravity.copy(), "Neg": by_gravity.copy()},
                                    "N": by_gravity.copy(), "V": by_gravity.copy()},
                        "Columns": {"M": c_gravity.copy(), "N": c_gravity.copy(), "V": c_gravity.copy()}}
        demands_gravity = {"x": gravity_temp, "y": gravity_temp}
        return demands_gravity

    def generate_initial_solutions(self, opt_sol, opt_modes, overstrength, table_sls):
        """
        Master file to run for each direction/iteration (only for 3D modelling)
        :param opt_sol: dict                        Dictionary containing structural element cross-section information
        :param opt_modes: dict                      Modal properties of the solution
        :param overstrength: float                         Overstrength factor
        :param table_sls: DataFrame                 DBD table at SLS
        :return: Design outputs and Demands on gravity (internal elements)
        """
        # Initialize demands on central/gravity structural elements
        demands_gravity = self.initialize_demands()

        # Initialize period to use (reset when running for Y direction)
        self.period_to_use = None

        # Generate initial solutions for both directions
        # For each primary direction of building, apply lateral loads and design the structural elements
        print("[PHASE] Target for MAFC...")
        cy, dy, spo2ida_data = self.target_for_mafc(opt_sol, overstrength)

        """Get action and demands"""
        print("[PHASE] Design structural components...")
        self.details = self.design_building(cy, dy, opt_sol, opt_modes, table_sls, demands_gravity)

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
            fd = CrossSectionSpace(self.data, None, None)
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

    def run_iterations(self, opt_sol, opt_modes, period_limits, table, maxiter=10, overstrength=1.0):
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
        # Initial assumptions for modal parameters retrieved from the optimal solution
        mstar = np.array([opt_sol["x_seismic"]["Mstar"], opt_sol["y_seismic"]["Mstar"]])
        part_factor = np.array([opt_sol["x_seismic"]["Part Factor"], opt_sol["y_seismic"]["Part Factor"]])
        modes = opt_modes["Modes"]

        # Initialize demands on central/gravity structural elements
        demands_gravity = self.initialize_demands()

        if isinstance(overstrength, float):
            overstrength = [overstrength] * 2

        # Retrieve necessary information from detailing results
        hinge_gr = self.details["gravity"]["hinge_models"]
        hinge_x = self.details["x"]["hinge_models"]
        hinge_y = self.details["y"]["hinge_models"]
        hinge_models = {"x_seismic": hinge_x, "y_seismic": hinge_y, "gravity": hinge_gr}
        cy = self.details["x"]["cy"]

        '''Iterate until all conditions are met'''
        # Count of iterations
        cnt = 0
        # Initialize SPO period
        spo_period = None
        # Reading from file the SPO parameters (initialize)
        read = True

        # Any warnings because of detailing?
        warnMax = self.details["x"]["warnMax"] or self.details["y"]["warnMax"] or \
                  self.details["gravity"]["warnings"]["warnMax"]

        # Initialize yield displacement and spo2ida results
        # guaranteed to be calculated at iteration 2, since iteration 1 always fails
        dy = None
        spo2ida_data = None

        # Start the iterations
        while (self.warnT or warnMax or not self.spo_validate or self.omegaWarn) and cnt + 1 <= maxiter:

            # Iterations skipping first iteration
            if (not self.spo_validate or self.omegaWarn or self.warnT) and cnt > 0:
                '''
                If SPO shape was not validated or
                if overstrength was not within the tolerable range or
                initial secant-to-yield period was not within the tolerable range
                '''
                # Rerun Modal analysis if warnT was raised
                print("[PHASE] Running Modal Analysis...")
                # Optimal solution might change, but the hinge_models cross-sections are not changed yet
                # It will be modified accordingly during design of the building
                periods, modes, part_factor, mstar, opt_sol = self.run_ma(opt_sol, hinge_models, period_limits, 0,
                                                                          spo_period=spo_period)

                # Target for MAFC
                print("[PHASE] Target MAFC in both directions...")
                cy, dy, spo2ida_data = self.target_for_mafc(opt_sol, overstrength, read=read)

                # Design the building
                print("[PHASE] Design and detail the building...")
                self.details = self.design_building(cy, dy, opt_sol, modes, table, demands_gravity)

                # Retrieve necessary information from detailing results
                hinge_models = {"x_seismic": self.details["x"]["hinge_models"],
                                "y_seismic": self.details["y"]["hinge_models"],
                                "gravity": self.details["gravity"]["hinge_models"]}

                # Correction if unsatisfactory detailing: modifying only towards increasing c-s
                # Direction X
                opt_sol = self.correct_due_to_unsatisfactory_detailing(opt_sol)

                # Any warnings because of detailing?
                warnMax = self.details["x"]["warnMax"] or self.details["y"]["warnMax"] or \
                          self.details["gravity"]["warnings"]["warnMax"]

                if warnMax:
                    hinge_models, warnMax = self.design_for_new_solution(cy, dy, opt_sol, modes, table, demands_gravity)

            """Create an OpenSees model and run static pushover - iterate analysis.
            Acts as the first estimation of secant to yield period. Second step after initial modal analysis.
            Here, the secant to yield period might vary from initial modal period. 
            Therefore, we need to use the former one."""
            # For a single frame assumed yield base shear
            vy_design = cy * part_factor * mstar * 9.81
            spo_pattern = np.round(modes, 2)
            print("[SPO] Starting SPO analysis...")
            spo_x, idealized_x, overstrength[0] = self.run_spo(opt_sol, hinge_models, vy_design[0], spo_pattern[:, 0],
                                                               overstrength[0], direction="x")
            spo_y, idealized_y, overstrength[1] = self.run_spo(opt_sol, hinge_models, vy_design[1], spo_pattern[:, 1],
                                                               overstrength[1], direction="y")
            for i in range(len(overstrength)):
                if overstrength[i] < 1.0:
                    overstrength[i] = 1.0

            # Calculate the period as secant to yield period
            spo_period = np.zeros((2, ))
            spo_period[0] = 2 * np.pi * np.sqrt(mstar[0] / (idealized_x[1][1] / idealized_x[0][1]))
            spo_period[1] = 2 * np.pi * np.sqrt(mstar[1] / (idealized_y[1][1] / idealized_y[0][1]))
            self.period_to_use = spo_period

            # Record OpenSees outputs
            self.model_outputs = {"MA": {"T": spo_period, "modes": spo_pattern, "gamma": part_factor, "mstar": mstar},
                                  "SPO": {"x": spo_x, "y": spo_y},
                                  "SPO_idealized": {"x": idealized_x, "y": idealized_y}}

            # Any warnings related to secant-to-yield period?
            self._any_period_warnings(spo_period, period_limits)

            # Reading SPO parameters from file (Set to False, as the actual shape is already identified)
            read = False

            print("[SUCCESS] Static pushover analysis was successfully performed.")

            print(f"[ITERATION {cnt + 1} END]")

            # Increase count of iterations
            cnt += 1

        if cnt == maxiter:
            print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": part_factor, "Mstar": mstar, "Period range": period_limits,
                         "overstrength": overstrength, "cy": cy, "dy": dy, "spo2ida": self.spo_shape}

        return ipbsd_outputs, spo2ida_data, opt_sol, modes, self.details, hinge_models, self.model_outputs

    def correct_due_to_unsatisfactory_detailing(self, opt_sol):
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
        return opt_sol

    def design_for_new_solution(self, cy, dy, opt_sol, modes, table, demands_gravity):
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
        return hinge_models, warnMax

    def _any_period_warnings(self, spo_period, period_limits):
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
