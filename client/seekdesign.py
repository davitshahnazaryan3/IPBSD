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


class SeekDesign:
    def __init__(self, ipbsd, sols, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath, gravity_loads):
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
        # Period to be used when performing important calculations
        self.period_to_use = None

    def target_for_mafc(self, solution, omega, read=True):

        # Overstrength
        overstrength = [omega] * 2

        # Set period equal to the actual period computed from MA or SPO analysis
        if self.period_to_use is not None:
            # Initial secant to yield period from SPO analysis
            period = self.period_to_use
        else:
            Tx = round(float(solution["x_seismic"]['T']), 1)
            Ty = round(float(solution["y_seismic"]['T']), 1)
            period = [Tx, Ty]

        if read:
            # Reads the input assumption if necessary
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

        # Get geometric mean of Cy
        cy = geo_mean(cy_xy)

        return cy, dy_xy, spo2ida_data

    def run_analysis(self, hinge=None):
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


    def design_building(self, cy, dy, sa, period_range, solution, modes, table_sls, rerun=False, gravity_demands=None,
                        detail_gravity=True):
        # Initialize hinge models
        hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

        # Get the acting lateral forces based on cy for each direction
        forces = {"x": self.ipbsd.get_action(solution["x_seismic"], cy, pd.DataFrame.from_dict(table_sls["x"]),
                                             self.gravity_loads, self.analysis_type),
                  "y": self.ipbsd.get_action(solution["y_seismic"], cy, pd.DataFrame.from_dict(table_sls["y"]),
                                             self.gravity_loads, self.analysis_type)}

        yield forces

        print(forces)
        sys.exit()

        yield

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

        # Initialize period to use (reset when running for Y direction)
        self.period_to_use = None

        # Generate initial solutions for both directions
        # For each primary direction of building, apply lateral loads and design the structural elements
        print("[PHASE] Commencing phase 3...")
        cy, dy, spo2ida_data = self.target_for_mafc(opt_sol, omega)

        """Get action and demands"""
        print("[PHASE] Commencing phase 4...")
        phase_4 = self.design_building(cy, dy, sa, period_range, opt_sol, opt_modes, table_sls,
                                       gravity_demands=demands_gravity, detail_gravity=False)

        forces = next(phase_4)
        demands, demands_gravity = next(phase_4)
        details, hinge_models, hinge_gravity, hard_ductility, fract_ductility = next(phase_4)
        warnMax, warnMin, warnings = next(phase_4)

