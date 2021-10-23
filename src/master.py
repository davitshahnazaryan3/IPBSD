"""
Runs the IPBSD framework
"""
from colorama import Fore
import numpy as np
import pandas as pd
import pickle

from src.crossSection import CrossSection
from src.crossSectionSpace import CrossSectionSpace
from src.designLimits import DesignLimits
from src.input import Input
from src.hazard import Hazard
from src.lossCurve import LossCurve
from src.periodRange import PeriodRange
from src.seekdesign import SeekDesign
from src.spectra import Spectra
from src.transformations import Transformations
from analysis.analysisMethods import run_opensees_analysis
from utils.ipbsd_utils import create_folder, export_results, initiate_msg, success_msg, error_msg, \
    create_and_export_cache, check_for_file
from utils.performance_obj_verifications import verify_period_range


class Master:
    def __init__(self, ipbsd):
        # IPBSD Main object
        self.ipbsd = ipbsd

        # Output path
        create_folder(self.ipbsd.output_path)

        # Outputs
        self.data = None            # IPBSD input object information
        self.mafe = None            # MAFE at each limit state (mean annual frequency of exceedance)
        self.true_hazard = None     # True hazard data
        self.period_limits = {}     # Period limits
        self.tables = None          # SLS table (DBD)
        self.combinations = None    # All section combinations
        self.opt_sol = None         # Optimal solutions

    def read_input(self):
        """
        Read input data
        :return: dict
                coefs: dict             Fitted hazard coefficients
                hazard_data: dict       Fitted hazard
                true_hazard: dict       True hazard
        """
        initiate_msg("---Starting IPBSD---")
        initiate_msg("---------------------------------")
        initiate_msg("Reading input arguments...")

        # Read the inputs to IPBSD
        data = Input(self.ipbsd.flag3d)
        data.read_inputs(self.ipbsd.input_filename)
        data.run_all()
        data.get_input_arguments()

        # Read hazard information
        hazard = Hazard(self.ipbsd.hazard_filename, self.ipbsd.output_path)
        coefs, hazard_data, self.true_hazard = hazard.read_hazard()

        # Get MAFE at each limit state
        mafe = hazard.get_mafe(coefs["PGA"], data.TR, "PGA")
        # Set MAFE of CLS to target MAFC
        mafe[-1] = self.ipbsd.target_mafc

        results = {
            'coefs': coefs,
            'hazard_data': hazard_data,
            'true_hazard': self.true_hazard,
        }

        # Update global variables
        self.data = data
        self.mafe = mafe

        if self.ipbsd.export:
            create_folder(self.ipbsd.output_path / "Cache")
            export_results(self.ipbsd.output_path / "Cache/input_cache", results, "pickle")

        success_msg("Input arguments have been successfully read!")

        print("...")

    def _get_loss_curve(self):
        lc = LossCurve(self.data.y, self.mafe, self.ipbsd.limit_eal)
        y_fit, mafe_fit = lc.get_loss_curve()
        lc.verify_eal()
        return lc.EAL, lc.y, y_fit, mafe_fit

    def _perform_transformations(self, theta_max, a_max):
        delta = np.zeros(theta_max.shape)
        alpha = np.zeros(a_max.shape)
        tables = {}

        for i in range(delta.shape[0]):
            transformations = Transformations(self.data, theta_max[i], a_max[i])
            table, _, deltas = transformations.generate_table()
            delta[i], alpha[i] = transformations.get_design_values(deltas)
            tag = "x" if i == 0 else "y"
            tables[tag] = table

        return tables, delta, alpha

    @staticmethod
    def _get_period_range(delta, alpha, sd, sa):
        pr = PeriodRange(delta, alpha, sd, sa)
        mew_sa, new_sd = pr.get_new_spectra()
        t_lower = pr.get_T_lower(mew_sa, new_sd)
        t_upper = pr.get_T_upper(mew_sa, new_sd)
        verify_period_range(t_lower, t_upper)
        return t_lower, t_upper

    def perform_calculations(self):
        """
        Runs IPBSD calculations
        :return:
        """
        # Stage 1
        initiate_msg("Starting IPBSD calculations...")

        eal, y, y_fit, mafe_fit = self._get_loss_curve()

        success_msg("Loss curve computed!\n...")

        # Stage 2
        initiate_msg("Computing engineering demand parameters as design limits...")
        dl = DesignLimits(self.ipbsd.slf_directory, self.data.y[1], self.data.nst, self.ipbsd.flag3d,
                          self.ipbsd.repl_cost, self.ipbsd.eal_correction, self.ipbsd.perform_scaling,
                          self.ipbsd.edp_profiles)
        dl.get_design_edps()

        if self.ipbsd.export:
            export_results(self.ipbsd.output_path / "Cache/SLFs", dl.SLFsCache, "pickle")
            export_results(self.ipbsd.output_path / "Cache/design_loss_contributions", dl.contributions, "pickle")

        success_msg("Design limits computed!\n...")

        if self.ipbsd.eal_correction:
            self.data.y[1] = dl.y
            # Performing EAL corrections to avoid overestimation of costs
            eal, y, y_fit, mafe_fit = self._get_loss_curve()

            success_msg(f"EAL corrections have been made, where the new EAL limit estimated to {eal:.2f}%\n...")

        create_and_export_cache(self.ipbsd.output_path / "Cache/lossCurve", "pickle",  y=self.data.y, mafe=self.mafe,
                                y_fit=y_fit, mafe_fit=mafe_fit, eal=eal, PLS=self.data.PLS)

        # Stage 3
        initiate_msg("Start transformation of design values into spectral coordinates...")
        self.tables, delta, alpha = self._perform_transformations(dl.theta_max, dl.a_max)

        if self.ipbsd.export:
            export_results(self.ipbsd.output_path / "Cache/table_sls", self.tables, "pickle")

        success_msg("Spectral values of design limits obtained!\n...")

        # Stage 4
        initiate_msg("Computing design spectrum at SLS...")
        spectra = Spectra()

        # spectral acceleration in g, and spectral displacement in %
        sa, sd = spectra.get_spectra(self.mafe[1], use_coefs=False, hazard=self.true_hazard)
        if self.ipbsd.export:
            try:
                shape = sa.shape[0]
                sls_spectrum = np.concatenate((spectra.T_RANGE.reshape(shape, 1),
                                               sd.reshape(shape, 1),
                                               sa.reshape(shape, 1)), axis=1)
                sls_spectrum = pd.DataFrame(data=sls_spectrum, columns=["Period", "Sd", "Sa"])
                export_results(self.ipbsd.output_path / "Cache/sls_spectrum", sls_spectrum, "csv")
            except:
                sls_spectrum = {"sa": sa, "sd": sd, "periods": spectra.T_RANGE}
                export_results(self.ipbsd.output_path / "Cache/sls_spectrum", sls_spectrum, "pickle")
        success_msg("Response spectrum at SLS generated!\n...")

        # Stage 5
        initiate_msg("Computing secant-to-yield period limits...")

        for i in range(delta.shape[0]):
            self.period_limits[f"{i+1}"] = self._get_period_range(delta[i], alpha[i], sd, sa)
            success_msg(f"Feasible period range identified: {self.period_limits[f'{i+1}']}")
        success_msg("...")

    def get_all_section_combinations(self):
        """
        Get all section combinations satisfying period bounds range
        Notes:
        * If solutions cache exists in outputs directory, the file will be read and an optimal solution based on
        least weight will be derived. This is done in order to avoid rerunning the heavy computations every single time,
        since the solutions file might not vary significantly or at all for each run, in case sensitivity tasks
        are being carried out
        * If solutions cache does not exist, then a solutions file will be created (could take some time depending on
        the complexity of the model) and then the optimal solution is derived.
        * If an optimal solution is provided, then eigenvalue analysis is performed for the optimal solution.
        No solutions cache will be derived.
        """
        # Solution file not provided
        if self.ipbsd.solution_file is None:
            # check whether solutions file was provided
            solution_x = check_for_file(self.ipbsd.solution_filex)
            solution_y = check_for_file(self.ipbsd.solution_filey)

            # Generates all possible section combinations assuming a stiffness reduction factor
            self.combinations = self._get_preliminary_structural_solutions(solution_x, solution_y)

        else:
            initiate_msg("Reading files containing initial section combinations satisfying period bounds...")
            # Solution file provided
            if isinstance(self.ipbsd.solution_file, str):
                # as a string
                with open(self.ipbsd.solution_file, "rb") as f:
                    self.opt_sol = pickle.load(f)
            else:
                # as a dict
                self.opt_sol = self.ipbsd.solution_file

        success_msg("Initial section combinations satisfying period bounds are obtained!\n...")

    def _get_preliminary_structural_solutions(self, solution_x, solution_y, iteration=False):
        def run_cross_section(period_limits, bays, path=None, iterate=False, perp=None):
            return CrossSection(self.data.nst, len(bays), self.data.fy, self.data.fc, bays,
                                self.data.heights, n_seismic, masses, self.ipbsd.fstiff, period_limits[0],
                                period_limits[1], export_directory=path, iteration=iterate, solution_perp=perp)

        def run_cross_section_space(period_limits, iterate=False):
            return CrossSectionSpace(self.data, period_limits, self.ipbsd.fstiff, iteration=iterate)

        if self.data.configuration == "perimeter" or not self.ipbsd.flag3d:
            # Get number of seismic frames and lumped masses along the height
            n_seismic, masses = self._get_system("x")
            if solution_x is None:
                cs = run_cross_section(self.period_limits["1"], self.data.spans_x,
                                       self.ipbsd.output_path / "Cache/solution_cache_x.csv")
                opt_sol, opt_modes = cs.find_optimal_solution()
                results_x = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

            elif solution_x is not None and not iteration:
                cs = run_cross_section(self.period_limits["1"], self.data.spans_x,
                                       self.ipbsd.output_path / "Cache/solution_cache_x.csv")
                opt_sol, opt_modes = cs.find_optimal_solution(solution_x)
                results_x = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

            else:
                cs = run_cross_section(self.period_limits["1"], self.data.spans_x, iterate=True)
                opt_sol, opt_modes = cs.find_optimal_solution(solution_x)
                results_x = {"opt_sol": opt_sol, "opt_modes": opt_modes}

            if self.ipbsd.flag3d:
                # Optimal solution in primary direction (dir1 or x)
                # Essentially the solution in Y direction will be selected by fixing the analysis columns
                n_seismic, masses = self._get_system("y")
                opt_sol_x = results_x["opt_sol"]

                if solution_y is None:
                    cs = run_cross_section(self.period_limits["2"], self.data.spans_y,
                                           self.ipbsd.output_path / "solution_cache_y.csv", perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution()
                    results_y = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

                elif solution_y is not None and not iteration:
                    cs = run_cross_section(self.period_limits["2"], self.data.spans_y,
                                           self.ipbsd.output_path / "solution_cache_y.csv", perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution(solution_y)
                    results_y = {"sols": cs.solutions, "opt_sol": opt_sol, "opt_modes": opt_modes}

                else:
                    cs = run_cross_section(self.period_limits["2"], self.data.spans_y, iterate=True,
                                           perp=opt_sol_x)
                    opt_sol, opt_modes = cs.find_optimal_solution(solution_y)
                    results_y = {"opt_sol": opt_sol, "opt_modes": opt_modes}

                return results_x, results_y
            else:
                return results_x

        else:
            # Space systems (3D only)
            if solution_x is None and self.data is not None:
                cs = run_cross_section_space(self.period_limits)
                cs.read_solutions(export_directory=self.ipbsd.output_path / "Cache/solution_space.csv")
                opt_sol_raw, opt_modes = cs.find_optimal_solution()
                # Convert optimal solution for usability. Gravity refers to central structural elements.
                # (e.g. transform it into a dictionary with keys: x_seismic, y_seismic, gravity)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}

            elif solution_x is not None:
                cs = run_cross_section_space(self.period_limits)
                cs.read_solutions(export_directory=self.ipbsd.output_path / "Cache/solution_space.csv")
                opt_sol_raw, opt_modes = cs.find_optimal_solution(solution_x)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}

            else:
                cs = run_cross_section_space(self.period_limits, iterate=True)
                cs.read_solutions()
                opt_sol_raw, opt_modes = cs.find_optimal_solution(solution_x)
                opt_sol = cs.get_section(opt_sol_raw)
                results = {"opt_sol_raw": opt_sol_raw, "opt_modes": opt_modes, "opt_sol": opt_sol, "sols": cs.solutions}

            return results

    def _get_system(self, direction):
        if self.data.configuration == "perimeter":
            n_seismic = 2
            masses = self.data.masses
        else:
            q_floor = self.data.i_d['bldg_ch'][0]
            q_roof = self.data.i_d['bldg_ch'][1]
            n_seismic = 1
            # Considering the critical frame only (with max tributary length)
            if direction == "x":
                # Spans in the primary direction
                spans = self.data.spans_x
                # Spans in the orthogonal direction
                spans_ort = np.array(self.data.spans_y)
            else:
                spans = self.data.spans_y
                spans_ort = np.array(self.data.spans_x)

            # Get the tributary lengths for the mass
            dist = np.convolve(spans_ort, np.ones(2), "valid") / 2
            tributary = max(dist)
            # Get the mass per storey
            masses = np.zeros((self.data.nst,))
            for st in range(self.data.nst):
                if st == self.data.nst - 1:
                    masses[st] = q_roof * sum(spans) * tributary / 9.81
                else:
                    masses[st] = q_floor * sum(spans) * tributary / 9.81

        return n_seismic, masses

    def perform_iterations(self):

        initiate_msg("Starting iterations for confined design!")

        if self.ipbsd.flag3d and self.data.configuration == "perimeter":
            # TODO, perimeter 3D incomplete
            # 3D building with perimeter seismic frames
            # gravity loads are calculated automatically (assuming a uniform distribution) - trapezoidal is more
            # appropriate
            gravity_loads = None

            if not self.ipbsd.gravity_cs:
                # Gravity solution not provided, create a generic one
                csg = {}

                for i in range(1, self.data.nst + 1):
                    column_size = 0.4 if i <= self.data.nst / 2 else 0.35
                    csg[f"hi{i}"] = [column_size]
                    csg[f"bx{i}"] = column_size
                    csg[f"hx{i}"] = 0.6
                    csg[f"by{i}"] = column_size
                    csg[f"hy{i}"] = 0.6
                csg = pd.DataFrame.from_dict(csg, orient='columns').iloc[0]
            else:
                # Gravity solution provided
                csg = pd.read_csv(self.ipbsd.gravity_cs, index_col=0).iloc[0]
        else:
            if self.ipbsd.flag3d:
                # Space system
                # gravity loads are calculated automatically
                gravity_loads = None
                self.period_limits = {"x": self.period_limits["1"], "y": self.period_limits["2"]}
                table_sls = self.tables

                # Retrieve optimal solutions
                if not self.opt_sol:
                    self.opt_sol = self.combinations["opt_sol"]
                    opt_modes = self.combinations["opt_modes"]

                else:
                    # Run modal analysis and identify the modal parameters
                    periods, modalShape, part_factor, mstar = \
                        run_opensees_analysis(1, self.opt_sol, None, self.data, None, self.ipbsd.fstiff,
                                              self.ipbsd.flag3d)

                    self.opt_sol["x_seismic"]["T"] = periods[0]
                    self.opt_sol["x_seismic"]["Part Factor"] = part_factor[0]
                    self.opt_sol["x_seismic"]["Mstar"] = mstar[0]
                    self.opt_sol["y_seismic"]["T"] = periods[1]
                    self.opt_sol["y_seismic"]["Part Factor"] = part_factor[1]
                    self.opt_sol["y_seismic"]["Mstar"] = mstar[1]

                    opt_modes = {"Periods": periods, "Modes": modalShape}

                self._seek_design_solution(gravity_loads, opt_modes, table_sls)

            else:
                # 2D frame assuming perimeter seismic frames
                gravity_loads = self.data.w_seismic
                period_limits = self.period_limits["1"]
                table_sls = self.tables["x"]
                # TODO, 2D incomplete

        success_msg("Iterations completed successfully!")
        success_msg("----------------------------------")
        success_msg("IPBSD concluded!!!")

    def _seek_design_solution(self, gravity_loads, modes, table):
        """
        Note: stiffness based off first yield point, at nominal point the stiffness is actually lower, and
        might act as a more realistic value. Notably Haselton, 2016 limits the secant yield stiffness between
        0.2EIg and 0.6EIg.
        """
        seek = SeekDesign(self.ipbsd.spo_filename, self.ipbsd.target_mafc, self.ipbsd.analysis_type, self.ipbsd.damping,
                          self.ipbsd.num_modes, self.ipbsd.fstiff, self.ipbsd.rebar_cover, gravity_loads,
                          self.data.configuration, self.data, self.true_hazard, self.ipbsd.output_path)

        seek.generate_initial_solutions(self.opt_sol, modes, self.ipbsd.overstrength, table)
        outputs = seek.run_iterations(self.opt_sol, modes, self.period_limits, table, self.ipbsd.maxiter,
                                      self.ipbsd.overstrength)

        ipbsd_outputs, spo_results, opt_sol, modes, details, hinges, model_outputs = outputs

        # Export cache
        if self.ipbsd.export:
            export_results(self.ipbsd.output_path / "Cache/spoAnalysisCurveShape", spo_results, "pickle")
            export_results(self.ipbsd.output_path / "Cache/optimal_solution", opt_sol, "pickle")
            export_results(self.ipbsd.output_path / "Cache/modes", modes, "pickle")
            export_results(self.ipbsd.output_path / "Cache/ipbsd", ipbsd_outputs, "pickle")
            export_results(self.ipbsd.output_path / "Cache/details", details, "pickle")
            export_results(self.ipbsd.output_path / "Cache/hinge_models", hinges, "pickle")
            export_results(self.ipbsd.output_path / "Cache/modelOutputs", model_outputs, "pickle")
