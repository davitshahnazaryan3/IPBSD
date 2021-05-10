"""
Runs the master file for Integrated Performance-Based Seismic Design (IPBSD)
"""
import sys
import timeit
import os
import json
import pandas as pd
import numpy as np
import pickle
from client.master import Master
from client.iterations import Iterations
from client.seekdesign import SeekDesign
from pathlib import Path


class IPBSD:
    def __init__(self, input_file, hazard_file, slfDir, spo_file, limit_eal, target_mafc, outputPath, analysis_type=1,
                 damping=.05, num_modes=3, iterate=False, maxiter=20, fstiff=0.5, rebar_cover=0.03,
                 flag3d=False, solutionFileX=None, solutionFileY=None, export_cache=False, holdFlag=False,
                 overstrength=None, replCost=None, gravity_cs=None, eal_correction=True, perform_scaling=True,
                 solutionFile=None):
        """
        Initializes IPBSD
        :param input_file: str              Input filename as '*.csv'
        :param hazard_file: str             Hazard filename as '*.pkl' or '*.pickle'
        :param slfDir: str                  Directory of SLFs derived via SLF Generator
        :param spo_file: str                SPO filename '*.csv'
        :param limit_eal: float             Liming value of EAL
        :param target_mafc: float           Target value of MAFC
        :param outputPath: str              Outputs path (where to store the outputs)
        :param analysis_type: int           Analysis type:
                                            1: Simplified ELF - no analysis is run, calculations based on
                                            simplified expressions, actions based on 1st mode shape (default)
                                            2: ELF - equivalent lateral force method of analysis, actions based on
                                            1st mode shape
                                            3: ELF & gravity - analysis under ELF and gravity loads
                                            4: RMSA - response method of spectral analysis, actions based on "n" modal
                                            shapes
                                            5: RMSA & gravity - analysis under RMSA and gravity loads
        :param damping: float               Ratio of critical damping
        :param num_modes: int               Number of modes to consider for SRSS (for analysis type 4 and 5)
        :param iterate: bool                Perform iterations or not (refers to iterations 4a and 3a)
        :param maxiter: int                 Maximum number of iterations for seeking a solution
        :param fstiff: float                Stiffness reduction factor
        :param rebar_cover: float           Reinforcement cover in m
        :param solutionFileX: str           Path to solution file to be used for design in X direction
        :param solutionFileY: str           Path to solution file to be used for design in Y direction (for 3D)
        :param export_cache: bool           Whether to export cache at each major step into an outputs directory or not
        :param holdFlag: bool               Flag to stop the framework once the solution combinations have been computed
                                            This allows the user to subdivide the framework into two sections if it
                                            takes too long to find a solution. Alternatively, if holdFlag is False,
                                            full framework will be run, and if solution.csv at the outputDir already
                                            exists, then the framework will skip the step of its derivation.
                                            holdFlag=False makes sense if the framework has already generated the csv
                                            file containing all valid solutions.
        :param overstrength: float          Overstrength ratio, leave on default and the user will automatically assign
                                            a value
        :param replCost: float              Replacement cost of the entire building
        :param gravity_cs: str              Path to gravity solution (for 3D modelling)
        :param eal_correction: bool         Perform EAL correction
        :param perform_scaling: bool        Perform scaling of SLFs to replCost (the scaling should not matter)
        :param solutionFile: str            Solution file containing a dictionary for the Space System (*.pickle)
        """
        self.dir = Path.cwd()
        self.input_file = input_file
        self.hazard_file = hazard_file
        self.slfDir = slfDir
        self.spo_file = spo_file
        self.limit_EAL = limit_eal
        self.target_mafc = target_mafc
        self.outputPath = outputPath
        self.analysis_type = analysis_type
        self.num_modes = num_modes
        self.damping = damping
        self.iterate = iterate
        self.maxiter = maxiter
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.export_cache = export_cache
        self.holdFlag = holdFlag
        self.overstrength = overstrength
        self.replCost = replCost
        self.gravity_cs = gravity_cs
        self.solutionFileX = solutionFileX
        self.solutionFileY = solutionFileY
        self.eal_correction = eal_correction
        self.perform_scaling = perform_scaling
        self.solutionFile = solutionFile

        # 2d (False) means, that even if the SLFs are provided for the entire building, only 1 direction
        # (defaulting to dir1) will be considered. This also entails the use of non-dimensional components,
        # however, during loss assessment the non-dimensional factor will not be applied (i.e. nd_factor=1.0)
        self.flag3d = flag3d

        if self.export_cache:
            # Create a folder for 'Cache' if none exists
            self.create_folder(self.outputPath / "Cache")

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

    @staticmethod
    def create_folder(directory):
        """
        creates directory
        :param directory: str                   Directory to be created
        :return: None
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory. " + directory)

    def export_results(self, filepath, data, filetype):
        """
        Store results in the database
        :param filepath: str                            Filepath, e.g. "directory/name"
        :param data:                                    Data to be stored
        :param filetype: str                            Filetype, e.g. npy, json, pkl, csv
        :return: None
        """
        if filetype == "npy":
            np.save(f"{filepath}.npy", data)
        elif filetype == "pkl" or filetype == "pickle":
            with open(f"{filepath}.pickle", 'wb') as handle:
                pickle.dump(data, handle)
        elif filetype == "json":
            with open(f"{filepath}.json", "w") as json_file:
                json.dump(data, json_file)
        elif filetype == "csv":
            data.to_csv(f"{filepath}.csv", index=False)

    def cacheRCMRF(self, ipbsd, details, sol, demands, path):
        """
        Creates cache to be used by RCMRF
        :param ipbsd: object                            Master class
        :param details: dict                            Details of design
        :param sol: dict                                Optimal solution
        :param demands: dict                            Demands on structural components
        :param path: str                                Path to directory to export the files
        :return: None
        """
        # Number of storeys
        nst = len(ipbsd.data.i_d["h_storeys"])

        # Loads (the only parameter needed for 3D)
        q_floor = float(ipbsd.data.i_d["bldg_ch"][0])
        q_roof = float(ipbsd.data.i_d["bldg_ch"][1])

        # For 2D only
        spansY = np.array([ipbsd.data.i_d["spans_Y"][x] for x in ipbsd.data.i_d["spans_Y"]])
        distLength = spansY[0] / 2

        # Distributed loads
        distLoads = [q_floor * distLength, q_roof * distLength]

        # Point loads, for now will be left as zero
        pLoads = 0.0

        # Number of gravity frames
        if not self.flag3d:
            nGravity = len(spansY) - 1
        else:
            nGravity = 0

        # PDelta loads/ essentially loads going to the gravity frames (for 2D only)
        if nGravity > 0:
            pDeltaLoad = ipbsd.data.pdelta_loads
        else:
            pDeltaLoad = 0.0

        # Masses (actual frame mass is exported) (for 2D only)
        masses = np.array(ipbsd.data.masses)

        # Creating a DataFrame for loads
        loads = pd.DataFrame(columns=["Storey", "Pattern", "Load"])

        for st in range(1, nst + 1):

            load = distLoads[1] if st == nst else distLoads[0]

            loads = loads.append({"Storey": st,
                                  "Pattern": "distributed",
                                  "Load": load}, ignore_index=True)

            # Point loads will be left as zeros for now
            loads = loads.append({"Storey": st,
                                  "Pattern": "point internal",
                                  "Load": pLoads}, ignore_index=True)
            loads = loads.append({"Storey": st,
                                  "Pattern": "point external",
                                  "Load": pLoads}, ignore_index=True)

            # PDelta loads (for 2D only)
            if nGravity > 0:
                # Associated with each seismic frame
                load = pDeltaLoad[st-1] / ipbsd.data.n_seismic
                loads = loads.append({"Storey": st,
                                      "Pattern": "pdelta",
                                      "Load": load}, ignore_index=True)

            else:
                # Add loads as zero
                loads = loads.append({"Storey": st,
                                      "Pattern": "pdelta",
                                      "Load": pDeltaLoad}, ignore_index=True)

            # Masses (for 2D only)
            loads = loads.append({"Storey": st,
                                  "Pattern": "mass",
                                  "Load": masses[st - 1] / ipbsd.data.n_seismic}, ignore_index=True)

            # Area loads (for both 2D and 3D)
            q = q_roof if st == nst else q_floor
            loads = loads.append({"Storey": st,
                                  "Pattern": "q",
                                  "Load": q}, ignore_index=True)

        # Exporting action for use by a Modeler module
        """
        For a two-way slab assumption, load distribution will not be uniform.
        For now and for simplicity, total load over each directions is computed and then divided by global length to 
        assume a uniform distribution. """
        self.export_results(path / "action", loads, "csv")

        # Materials
        fc = ipbsd.data.i_d["fc"][0]
        fy = ipbsd.data.i_d["fy"][0]
        Es = ipbsd.data.i_d["Es"][0]

        # Elastic modulus of uncracked concrete
        Ec = (3320 * np.sqrt(fc) + 6900) * self.fstiff

        materials = pd.DataFrame({"fc": fc,
                                  "fy": fy,
                                  "Es": Es,
                                  "Ec": Ec}, index=[0])

        # Exporting the materials file for use by a Modeler module
        self.export_results(path / "materials", materials, "csv")

        # """Section properties (for the Haselton model)"""
        # def get_sections(i, sections, details, sol, demands, ele, iterator, st, bay, nbays, bayName):
        #     eleType = "Beam" if ele == "Beams" else "Column"
        #     pos = "External" if bay == 1 else "Internal"
        #     p0 = pos[0].lower() if eleType == "Column" else ""
        #
        #     # C-S dimensions
        #     if eleType == "Beam":
        #         b = sol.loc[f"b{st}"]
        #         h = sol.loc[f"h{st}"]
        #         Ptotal = 0.0
        #         length = spansX[bay - 1]
        #         MyPos = details[ele]["Pos"][i][3]["yield"]["moment"]
        #         MyNeg = details[ele]["Neg"][i][3]["yield"]["moment"]
        #         coverPos = details[ele]["Pos"][i][0]["cover"]
        #         coverNeg = details[ele]["Neg"][i][0]["cover"]
        #         roPos = details[ele]["Pos"][i][0]["reinforcement"] / (b * (h - coverPos))
        #         roNeg = details[ele]["Neg"][i][0]["reinforcement"] / (b * (h - coverNeg))
        #
        #     else:
        #         b = h = sol.loc[f"h{p0}{st}"]
        #         Ptotal = max(demands[ele]["N"][st - 1][bay - 1], demands[ele]["N"][st - 1][nbays - bay + 1], key=abs)
        #         length = heights[st - 1]
        #         MyPos = MyNeg = details[ele][i][3]["yield"]["moment"]
        #         coverPos = coverNeg = details[ele][i][0]["cover"]
        #         roPos = roNeg = details[ele][i][0]["reinforcement"] / (b * (h - coverPos))
        #
        #     # Residual strength of component
        #     res = iterator[i][3]["fracturing"]["moment"] / iterator[i][3]["yield"]["moment"]
        #
        #     # Appending into the DataFrame
        #     sections = sections.append({"Element": eleType,
        #                                 "Bay": bayName,
        #                                 "Storey": st,
        #                                 "Position": pos,
        #                                 "b": float(b),
        #                                 "h": float(h),
        #                                 "coverPos": float(coverPos),
        #                                 "coverNeg": float(coverNeg),
        #                                 "Ptotal": Ptotal,
        #                                 "MyPos": float(MyPos),
        #                                 "MyNeg": float(MyNeg),
        #                                 "asl": asl,
        #                                 "Ash": float(iterator[i][0]["A_sh"]),
        #                                 "spacing": float(iterator[i][0]["spacing"]),
        #                                 "db": db,
        #                                 "c": c,
        #                                 "D": D,
        #                                 "Res": float(res),
        #                                 "Length": length,
        #                                 "ro_long_pos": float(roPos),
        #                                 "ro_long_neg": float(roNeg)}, ignore_index=True)
        #     return sections

        # # Constants - assumptions - to be made more flexible
        # asl = 0
        # c = 1
        # D = 1
        # db = 20
        # nbays = len(spansX)
        #
        # # Initialize sections
        # sections = pd.DataFrame(columns=["Element", "Bay", "Storey", "Position", "b", "h", "coverPos", "coverNeg",
        #                                  "Ptotal", "MyPos", "MyNeg", "asl", "Ash", "spacing", "db", "c", "D", "Res",
        #                                  "Length", "ro_long_pos", "ro_long_neg"])
        #
        # for ele in details:
        #     if ele == "Beams":
        #         iterator = details[ele]["Pos"]
        #     else:
        #         iterator = details[ele]
        #
        #     for i in iterator:
        #         st = int(i[1])
        #         bay = int(i[3])
        #         sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bay, nbays, bayName=bay)
        #
        # # Add symmetric columns
        # if bay < nbays:
        #     bayName = bay
        #     for bb in range(nbays - bay + 1, 0, -1):
        #         bayName += 1
        #         for st in range(1, nst + 1):
        #             ele = "Columns"
        #             i = f"S{st}B{bb}"
        #             iterator = details[ele]
        #             sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bb, nbays, bayName)
        #
        #     bayName = bay
        #     for bb in range(nbays - bay, 0, -1):
        #         bayName += 1
        #         for st in range(1, nst + 1):
        #             ele = "Beams"
        #             i = f"S{st}B{bb}"
        #             iterator = details[ele]["Pos"]
        #             sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bb, nbays, bayName)
        #
        # # Exporting the sections file for use by Modeler module as Haselton springs
        # self.export_results(path / "haselton_springs", sections, "csv")

    def run_ipbsd(self, seekdesign=False):
        """
        Runs the framework
        :param seekdesign: bool                     TODO, currently supports 3D space systems
        :return:
        """
        """Calling the master file"""
        ipbsd = Master(self.dir, flag3d=self.flag3d)

        """Generating and storing the input arguments"""
        ipbsd.read_input(self.input_file, self.hazard_file, self.outputPath)

        if not self.flag3d and ipbsd.data.i_d["configuration"][0] == "space":
            print("[WARNING] It is recommended to run space systems with 3D modelling!")

        # Initiate {project name}
        print(f"[INITIATE] Starting IPBSD for {ipbsd.data.case_id}")

        print("[PHASE] Commencing phase 1...")
        self.create_folder(self.outputPath)
        ipbsd.data.i_d["MAFC"] = self.target_mafc
        ipbsd.data.i_d["EAL"] = self.limit_EAL
        # Store IPBSD inputs as a json
        if self.export_cache:
            self.export_results(self.outputPath / "Cache/input_cache", ipbsd.data.i_d, "json")
        print("[SUCCESS] Input arguments have been read and successfully stored")

        """Get EAL"""
        lam_ls = ipbsd.get_hazard_pga(self.target_mafc)
        eal, y_fit, lam_fit = ipbsd.get_loss_curve(lam_ls, self.limit_EAL)
        lossCurve = {"y": ipbsd.data.y, "lam": lam_ls, "y_fit": y_fit, "lam_fit": lam_fit, "eal": eal,
                     "PLS": ipbsd.data.PLS}

        """Get design limits"""
        theta_max, a_max, slfsCache, contributions = ipbsd.get_design_values(self.slfDir, self.replCost,
                                                                             self.eal_correction, self.perform_scaling)

        if self.export_cache:
            self.export_results(self.outputPath / "Cache/SLFs", slfsCache, "pickle")
            self.export_results(self.outputPath / "Cache/design_loss_contributions", contributions, "pickle")
        print("[SUCCESS] SLF successfully read, and design limits are calculated")

        if self.eal_correction:
            # Performing EAL corrections to avoid overestimation of costs
            eal, y_fit, lam_fit = ipbsd.get_loss_curve(lam_ls, self.limit_EAL)
            print(f"[SUCCESS] EAL corrections have been made, where the new EAL limit estimated to {eal:.2f}%")
            lossCurve = {"y": ipbsd.data.y, "lam": lam_ls, "y_fit": y_fit, "lam_fit": lam_fit, "eal": eal,
                         "PLS": ipbsd.data.PLS}

        """Transform design values into spectral coordinates"""
        tables, delta_spectral, alpha_spectral = ipbsd.perform_transformations(theta_max, a_max)
        if self.export_cache:
            self.export_results(self.outputPath / "Cache/table_sls", tables, "pickle")
        print("[SUCCESS] Spectral values of design limits are obtained")

        """Get spectra at SLS"""
        print("[PHASE] Commencing phase 2...")
        sa, sd, period_range = ipbsd.get_spectra(lam_ls[1])
        if self.export_cache:
            try:
                i = sa.shape[0]
                sls_spectrum = np.concatenate((period_range.reshape(i, 1), sd.reshape(i, 1), sa.reshape(i, 1)), axis=1)
                sls_spectrum = pd.DataFrame(data=sls_spectrum, columns=["Period", "Sd", "Sa"])
                self.export_results(self.outputPath / "Cache/sls_spectrum", sls_spectrum, "csv")
            except:
                sls_spectrum = {"sa": sa, "sd": sd, "periods": period_range}
                self.export_results(self.outputPath / "Cache/sls_spectrum", sls_spectrum, "pickle")

        print("[SUCCESS] Response spectra generated")

        """Get feasible fundamental period range"""
        period_limits = {}
        for i in range(delta_spectral.shape[0]):
            period_limits[f"{i+1}"] = ipbsd.get_period_range(delta_spectral[i], alpha_spectral[i], sd, sa)
            print(f"[SUCCESS] Feasible period range identified: {period_limits[f'{i+1}']}")

        """Get all section combinations satisfying period bounds range
        Notes: 
        * If solutions cache exists in outputs directory, the file will be read and an optimal solution based on
        least weight will be derived.
        * If solutions cache does not exist, then a solutions file will be created (may take some time) and then the
        optimal solution is derived.
        * If an optimal solution is provided, then eigenvalue analysis is performed for the optimal solution. 
        No solutions cache will be derived.
        """
        # Check whether solutions file was provided
        if not isinstance(self.solutionFileX, int):
            if self.solutionFileX is not None:
                try:
                    solution_x = pd.read_csv(self.solutionFileX)
                except:
                    solution_x = None
            else:
                solution_x = None
        else:
            solution_x = self.solutionFileX

        if not isinstance(self.solutionFileY, int):
            if self.solutionFileY is not None:
                try:
                    solution_y = pd.read_csv(self.solutionFileY)
                except:
                    solution_y = None
            else:
                solution_y = None
        else:
            solution_y = self.solutionFileY

        # Generates all possible section combinations assuming a stiffness reduction factor
        if self.solutionFile is None:
            results = ipbsd.get_all_section_combinations(period_limits, fstiff=self.fstiff, data=ipbsd.data,
                                                         cache_dir=self.outputPath/"Cache",
                                                         solution_x=solution_x, solution_y=solution_y)
            print("[SUCCESS] All section combinations were identified")
        else:
            if isinstance(self.solutionFile, str):
                with open(self.solutionFile, "rb") as f:
                    opt_sol = pickle.load(f)
            else:
                opt_sol = self.solutionFile

        # The main portion of IPBSD is completed. Now corrections should be made for the assumptions
        if not self.holdFlag:

            if self.flag3d and ipbsd.data.configuration == "perimeter":
                # To be calculated automatically (assuming a uniform distribution) - trapezoidal is more appropriate
                gravity_loads = None
                # Gravity solution not necessary for space systems
                if self.gravity_cs is None:
                    # In case gravity solution was not provided, create a generic one; for 3D modelling only
                    csg = {}
                    for i in range(1, ipbsd.data.nst + 1):
                        column_size = 0.4 if i <= ipbsd.data.nst / 2 else 0.35
                        csg[f"hi{i}"] = [column_size]
                        csg[f"bx{i}"] = column_size
                        csg[f"hx{i}"] = 0.6
                        csg[f"by{i}"] = column_size
                        csg[f"hy{i}"] = 0.6
                    csg = pd.DataFrame.from_dict(csg, orient='columns').iloc[0]
                else:
                    csg = pd.read_csv(self.gravity_cs, index_col=0).iloc[0]

                # Prepare the input variables to be passed to Iterations for 3D modelling
                opt_sol = {"x_seismic": results[0]["opt_sol"], "y_seismic": results[1]["opt_sol"], "gravity": csg}
                sols = {"x": results[0]["sols"], "y": results[1]["sols"]}
                opt_modes = {"x": results[0]["opt_modes"], "y": results[1]["opt_modes"]}
                period_limits = {"x": period_limits["1"], "y": period_limits["2"]}
                table_sls = tables

            else:
                if self.flag3d:
                    # Space systems
                    # It is best to automatically calculate the gravity loads based on global dimensions
                    gravity_loads = None
                    period_limits = {"x": period_limits["1"], "y": period_limits["2"]}
                    table_sls = tables
                else:
                    # 2D frames
                    period_limits = period_limits["1"]
                    gravity_loads = ipbsd.data.w_seismic
                    table_sls = tables["x"]

                # Just the direction of interest (modelling a 2D frame, without the perpendicular frame or
                # gravity frames
                # For 3D space systems opt_sol includes information on all structural elements
                if "results" in locals():
                    opt_sol = results["opt_sol"]
                    opt_modes = results["opt_modes"]
                    # Solutions file is not necessary in SeekDesign
                    sols = results["sols"]
                else:
                    # Run Modal Analysis and identify the modal parameters
                    model_periods, modalShape, gamma, mstar = ipbsd.ma_analysis(opt_sol, None, None, self.fstiff)
                    # Initial solution was provided
                    opt_sol["x_seismic"]["T"] = model_periods[0]
                    opt_sol["x_seismic"]["Part Factor"] = gamma[0]
                    opt_sol["x_seismic"]["Mstar"] = mstar[0]

                    opt_sol["y_seismic"]["T"] = model_periods[1]
                    opt_sol["y_seismic"]["Part Factor"] = gamma[1]
                    opt_sol["y_seismic"]["Mstar"] = mstar[1]

                    opt_modes = {"Periods": model_periods, "Modes": modalShape}
                    # Update tee
                    sols = None

            if seekdesign and self.flag3d:
                seek = SeekDesign(ipbsd, self.spo_file, self.target_mafc, self.analysis_type, self.damping,
                                  self.num_modes, self.fstiff, self.rebar_cover, self.outputPath,
                                  gravity_loads=gravity_loads)

                # Initial design solutions of all structural elements based on elastic analysis
                details = seek.generate_initial_solutions(opt_sol, opt_modes, self.overstrength, table_sls)
                outputs = seek.run_iterations(opt_sol, opt_modes, period_limits, table_sls, maxiter=self.maxiter)

                ipbsd_outputs = outputs[0]
                spo_results = outputs[1]
                opt_sol = outputs[2]
                modes = outputs[3]
                details = outputs[4]
                hinges = outputs[5]
                model_outputs = outputs[6]

                # Export outputs and cache
                if self.export_cache:
                    """Creating DataFrames to store for RCMRF input"""
                    self.cacheRCMRF(ipbsd, None, None, None, path=self.outputPath)

                    # Exporting the IPBSD outputs
                    self.create_folder(self.outputPath / f"Cache")
                    self.export_results(self.outputPath / f"Cache/lossCurve", lossCurve, "pickle")
                    self.export_results(self.outputPath / f"Cache/spoAnalysisCurveShape", spo_results, "pickle")
                    self.export_results(self.outputPath / f"Cache/optimal_solution", opt_sol, "pickle")
                    self.export_results(self.outputPath / f"Cache/modes", modes, "pickle")
                    self.export_results(self.outputPath / f"Cache/ipbsd", ipbsd_outputs, "pickle")
                    self.export_results(self.outputPath / f"Cache/details", details, "pickle")
                    self.export_results(self.outputPath / f"Cache/hinge_models", hinges, "pickle")
                    self.export_results(self.outputPath / f"Cache/modelOutputs", model_outputs, "pickle")

            else:
                # Call the iterations function (iterations have not yet started though)
                iterations = Iterations(ipbsd, sols, self.spo_file, self.target_mafc, self.analysis_type, self.damping,
                                        self.num_modes, self.fstiff, self.rebar_cover, self.outputPath,
                                        gravity_loads=gravity_loads, flag3d=self.flag3d)

                # Initiate design of the entire building / frame
                if self.flag3d:
                    # Initial design solutions of all structural elements based on elastic analysis
                    init_design, demands_gravity = iterations.generate_initial_solutions(opt_sol, opt_modes,
                                                                                         self.overstrength, sa,
                                                                                         period_range, table_sls)

                    outputs = iterations.run_iterations_for_3d(init_design, demands_gravity, period_limits, opt_sol,
                                                               opt_modes, sa, period_range, table_sls,
                                                               iterate=self.iterate, maxiter=self.maxiter,
                                                               omega=self.overstrength)
                    frames = ["x", "y"]
                    ipbsd_outputs = outputs["ipbsd_outputs"]
                    spoResults = outputs["spoResults"]
                    opt_sol = outputs["opt_sol"]
                    demands = outputs["demands"]
                    details = outputs["details"]
                    hinge_models = outputs["hinge_models"]
                    action = outputs["action"]
                    modelOutputs = outputs["modelOutputs"]

                else:
                    # Run the validations and iterations if need be
                    ipbsd_outputs, spoResults, opt_sol, demands, details, hinge_models, action, modelOutputs = \
                        iterations.validations(opt_sol, opt_modes, sa, period_range, table_sls, period_limits,
                                               self.iterate, self.maxiter, omega=self.overstrength)
                    frames = ["x"]
                    outputs = None

                """Iterations are completed and IPBSD is finalized"""
                # Export main outputs and cache
                if self.export_cache:
                    for i in frames:
                        if self.flag3d:
                            spoResults_to_store = spoResults[i]
                            opt_sol_to_store = opt_sol[i]
                            action_to_store = action[i]
                            demands_to_store = demands[i]
                            ipbsd_outputs_to_store = ipbsd_outputs[i]
                            details_to_store = details[i]
                            hinge_models_to_store = hinge_models[i][f"{i}_seismic"]
                            modelOutputs_to_store = modelOutputs[i]
                            """Creating DataFrames to store for RCMRF input"""
                            self.cacheRCMRF(ipbsd, details_to_store, opt_sol_to_store[f"{i}_seismic"], demands_to_store,
                                            path=self.outputPath)
                            self.export_results(self.outputPath / f"Cache/gravity_hinges", hinge_models[i]["gravity"],
                                                "csv")
                        else:
                            spoResults_to_store = spoResults
                            opt_sol_to_store = opt_sol
                            action_to_store = action
                            demands_to_store = demands
                            ipbsd_outputs_to_store = ipbsd_outputs
                            details_to_store = details
                            hinge_models_to_store = hinge_models
                            modelOutputs_to_store = modelOutputs
                            self.cacheRCMRF(ipbsd, details_to_store, opt_sol_to_store, demands_to_store,
                                            path=self.outputPath)

                        """Storing the outputs"""
                        # Exporting the IPBSD outputs
                        self.create_folder(self.outputPath / f"Cache/frame{i}")
                        self.export_results(self.outputPath / f"Cache/frame{i}/lossCurve", lossCurve, "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/spoAnalysisCurveShape", spoResults_to_store,
                                            "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/optimal_solution", opt_sol_to_store,
                                            "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/action", action_to_store, "csv")
                        self.export_results(self.outputPath / f"Cache/frame{i}/demands", demands_to_store, "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/ipbsd", ipbsd_outputs_to_store, "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/details", details_to_store, "pickle")
                        self.export_results(self.outputPath / f"Cache/frame{i}/hinge_models", hinge_models_to_store, "csv")
                        self.export_results(self.outputPath / f"Cache/frame{i}/modelOutputs", modelOutputs_to_store,
                                            "pickle")

                print("[SUCCESS] Structural elements were designed and detailed. SPO curve parameters were estimated")
                # Note: stiffness based off first yield point, at nominal point the stiffness is actually lower, and
                # might act as a more realistic value. Notably Haselton, 2016 limits the secant yield stiffness between
                # 0.2EIg and 0.6EIg.

        print("[END] IPBSD was performed successfully")


if __name__ == "__main__":

    """
    :param analysis_type: int                   Analysis type:
                                                1: Simplified ELF - no analysis is run, calculations based on 
                                                simplified expressions
                                                2: ELF - equivalent lateral force method of analysis
                                                3: ELF & gravity - analysis under ELF and gravity loads
                                                4: RMSA - response method of spectral analysis
                                                5: RMSA & gravity - analysis under RMSA and gravity loads
    :param input_file: str                      Input file as '*.csv'
    :param hazard_file: str                     Hazard file as '*.pkl' or '*.pickle'
    :param slf_file: str                        File containing SLF curves as '*.xlsx'
    :param spo_file: str                        File containing SPO curve assumptions as '*.csv'
    :param limit_eal: float                     Limit EAL performance objective
    :param mafc_target: float                   MAFC target performance objective
    :param damping: float                       Damping (e.g. 0.05)
    :param system: str                          Lateral load resisting system type (Perimeter or Space)
    :param maxiter: int                         Maximum number of iterations for detailing optimization
    :param fstiff: float                        Section stiffness reduction factor
    """
    # Paths
    path = Path.cwd()
    outputPath = path.parents[0] / ".applications/LOSS Validation Manuscript/Case21"

    # Add input arguments
    analysis_type = 3
    input_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Case21/ipbsd_input.csv"
    hazard_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Case21/Hazard-LAquila-Soil-C.pkl"
    slfDir = outputPath / "slfoutput"
    spo_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Case21/spo.csv"
    gravity_cs = None
    limit_eal = 1.0
    mafc_target = 2.e-4
    damping = .05
    maxiter = 10
    fstiff = 0.5
    overstrength = 1.0
    export_cache = True
    holdFlag = False
    iterate = True
    solutionFileX = None
    solutionFileY = None
    flag3d = True
    replCost = 349459.2

    # Design solution to use (leave None, if the tool needs to look for the solution)
    # solutionFile = path.parents[0] / ".applications/case1/designSol.csv"
    solutionFile = None

    method = IPBSD(input_file, hazard_file, slfDir, spo_file, limit_eal, mafc_target, outputPath, analysis_type,
                   damping=damping, num_modes=2, iterate=iterate, maxiter=maxiter, fstiff=fstiff,
                   flag3d=flag3d, solutionFileX=solutionFileX, solutionFileY=solutionFileY, export_cache=export_cache,
                   holdFlag=holdFlag, overstrength=overstrength, rebar_cover=0.03, replCost=replCost,
                   gravity_cs=gravity_cs)

    start_time = method.get_init_time()
    method.run_ipbsd()
    method.get_time(start_time)

    # import sys
    # sys.exit()
