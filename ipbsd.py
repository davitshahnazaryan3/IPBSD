"""
Runs the master file for IPBSD
"""
import timeit
import os
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from client.master import Master


class IPBSD:
    def __init__(self, input_file, hazard_file, slfDir, spo_file, limit_eal, target_mafc, outputPath, analysis_type=1,
                 damping=.05, num_modes=3, record=True, iterate=False, system="Perimeter", maxiter=20, fstiff=0.5,
                 rebar_cover=0.03, geometry="2D"):
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
        :param record: bool                 Flag for storing the results
        :param iterate: bool                Perform iterations or not (refers to iterations 4a and 3a)
        :param system: str                  Structural system of the building (Perimeter or Space)
        :param maxiter: int                 Maximum number of iterations for seeking a solution
        :param fstiff: float                Stiffness reduction factor
        :param rebar_cover: float           Reinforcement cover in m
        :param geometry: str                "2d" if a single frame is considered, "3d" if a full building is considered
        """
        self.dir = Path.cwd()
        self.input_file = input_file
        self.hazard_file = hazard_file
        self.slfDir = slfDir
        self.spo_file = spo_file
        self.limit_EAL = limit_eal
        self.target_MAFC = target_mafc
        self.outputPath = outputPath
        self.analysis_type = analysis_type
        self.num_modes = num_modes
        self.damping = damping
        self.record = record
        self.iterate = iterate
        self.system = system                # TODO, currently it does nothing, IPBSD is only working with Perimeter frames, Space (loads etc. to be updated) frames to be added
        self.maxiter = maxiter
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        # 2d means, that even if the SLFs are provided for the entire building, only 1 direction (defaulting to dir1)
        # will be considered. This also entails the use of non-dimensional components, however, during loss assessment
        # the non-dimensional factor will not be applied (i.e. nd_factor=1.0)
        if geometry.lower() == "2d":
            self.geometry = 0
        else:
            self.geometry = 1

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

    def create_folder(self, directory):
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

    def store_results(self, filepath, data, filetype):
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
            with open(f"{filepath}.pkl", 'wb') as handle:
                pickle.dump(data, handle)
        elif filetype == "json":
            with open(f"{filepath}.json", "w") as json_file:
                json.dump(data, json_file)
        elif filetype == "csv":
            data.to_csv(f"{filepath}.csv")

    def iterate_phase_3(self, ipbsd, case_directory, opt_sol, omega):
        """
        Runs phase 3 of the framework
        :param ipbsd: object                            Master class
        :param case_directory: str                      Directory of the case study application
        :param opt_sol: Series                          Solution containing c-s and modal properties
        :param omega: float                             Overstrength factor
        :return part_factor: float                      Participation factor of first mode
        :return m_star: float                           Effective first modal mass
        :return say: float                              Spectral acceleration at yield in g
        :return dy: float                               Spectral displacement at yield in m
        """
        """Estimate parameters for SPO curve and compare with assumed shape"""
        period = round(float(opt_sol['T']), 1)
        spo_data = ipbsd.data.initial_spo_data(period, self.dir / "client" / self.spo_file)
        spo2ida_data = ipbsd.perform_spo2ida(spo_data)
        if self.record:
            self.store_results(case_directory / "spo2ida_results", spo2ida_data, "pkl")
        print("[SUCCESS] SPO2IDA was performed")

        """Yield strength optimization for MAFC and verification"""
        part_factor = opt_sol["Part Factor"]
        m_star = opt_sol["Mstar"]
        say, dy = ipbsd.verify_mafc(period, spo2ida_data, part_factor, self.target_MAFC, omega, hazard="True")
        print("[SUCCESS] MAFC was validated")
        return part_factor, m_star, say, dy

    def iterate_phase_4(self, ipbsd, say, dy, sa, period_range, solution, modes, table_sls, t_lower, t_upper):
        """
        Runs phase 4 of the framework
        :param ipbsd: object                            Master class
        :param say: float                               Spectral acceleration at yield in g
        :param dy: float                                Spectral displacement at yield in m
        :param sa: list                                 Spectral accelerations in g of the spectrum
        :param period_range: list                       Periods of the spectrum
        :param solution: Series                         Solution containing c-s and modal properties
        :param modes: dict                              Periods and normalized modal shapes of the solution
        :param table_sls: DataFrame                     Table with SLS parameters
        :param t_lower: float                           Lower period limit
        :param t_upper: float                           Upper period limit
        :return forces: DataFrame                       Acting forces
        :return demands: dict                           Demands on the structural elements
        :return details: dict                           Designed element properties from the moment-curvature
                                                        relationship
        :return hard_ductility: float                   Estimated system hardening ductility
        :return fract_ductility: float                  Estimated system fracturing ductility
        :return warn: bool                              Indicates whether any warnings were displayed
        :return warnings: dict                          Dictionary of boolean warnings for each structural element
        """
        if self.analysis_type == 4 or self.analysis_type == 5:
            se_rmsa = ipbsd.get_sa_at_period(say, sa, period_range, modes["Periods"])
        else:
            se_rmsa = None

        self.num_modes = min(self.num_modes, ipbsd.data.nst)
        if self.analysis_type == 4 or self.analysis_type == 5:
            corr = ipbsd.get_correlation_matrix(modes["Periods"], self.num_modes, damping=self.damping)
        else:
            corr = None
        forces = ipbsd.get_action(solution, say, pd.DataFrame.from_dict(table_sls),
                                  ipbsd.data.w_seismic, self.analysis_type, self.num_modes, modes, modal_sa=se_rmsa)

        print("[SUCCESS] Actions on the structure for analysis were estimated")
        yield forces

        """Perform ELF analysis"""
        if self.analysis_type == 1:
            demands = ipbsd.run_muto_approach(solution, list(forces["Fi"]), ipbsd.data.h, ipbsd.data.spans_x)
        elif self.analysis_type == 2:
            demands = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), fstiff=self.fstiff)
        elif self.analysis_type == 3:
            demands = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                         fstiff=self.fstiff)
        elif self.analysis_type == 4 or self.analysis_type == 5:
            demands = {}
            for mode in range(self.num_modes):
                demands[f"Mode{mode+1}"] = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"][mode]),
                                                              fstiff=self.fstiff)
            demands = ipbsd.perform_cqc(corr, demands)
            if self.analysis_type == 5:
                demands_gravity = ipbsd.run_analysis(self.analysis_type, solution, grav_loads=list(forces["G"]),
                                                     fstiff=self.fstiff)
                # Combining gravity and RSMA results
                for eleType in demands_gravity.keys():
                    for dem in demands_gravity[eleType].keys():
                        demands[eleType][dem] = demands[eleType][dem] + demands_gravity[eleType][dem]

        else:
            raise ValueError("[EXCEPTION] Incorrect analysis type...")
        yield demands
        print("[SUCCESS] Analysis completed and demands on structural elements were estimated.")

        # todo, estimation of global peak to yield ratio to be added
        """Design the structural elements"""
        details, hard_ductility, fract_ductility, warn, warnings = ipbsd.design_elements(demands, solution, modes,
                                                                                         t_lower, t_upper, dy,
                                                                                         fstiff=self.fstiff,
                                                                                         cover=self.rebar_cover)
        yield details, hard_ductility, fract_ductility
        yield warn, warnings

    def seek_solution(self, data, warnings, sols):
        """
        Seeks a solution within the already generated section combinations file
        :param data: object                         Input arguments
        :param warnings: dict                       Dictionary of boolean warnings for each structural element
        :param sols: DataFrame                      Solution combos
        :return: Series                             Solution containing c-s and modal properties
        :return: dict                               Modes corresponding to the solution for RSMA
        """
        nst = data.nst
        opt_old = sols[sols["Weight"] == sols["Weight"].min()].iloc[0]
        opt = opt_old.copy().drop(["T", "Weight", "Mstar", "Part Factor"])

        # Increment for cross-section increments for elements with warnings
        increment = 0.05
        for ele in warnings["Columns"]:
            if warnings["Columns"][ele] == 1:
                # Increase section cross-section
                storey = ele[1]
                bay = int(ele[3])
                if bay == 1:
                    opt[f"he{storey}"] = opt[f"he{storey}"] + increment

                else:
                    opt[f"hi{storey}"] = opt[f"hi{storey}"] + increment

        for ele in warnings["Beams"]:
            storey = ele[1]
            if warnings["Beams"][ele] == 1:
                # Increase section cross-section
                opt[f"b{storey}"] = opt[f"he{storey}"]
                opt[f"h{storey}"] = opt[f"h{storey}"] + increment

        '''Enforce constraints'''
        # Column storey constraints
        for st in range(1, nst):
            if opt[f"he{st}"] > opt[f"he{st + 1}"] + 0.05:
                opt[f"he{st + 1}"] = opt[f"he{st}"] - 0.05
            if opt[f"he{st}"] < opt[f"he{st + 1}"]:
                opt[f"he{st}"] = opt[f"he{st + 1}"]

            if opt[f"hi{st}"] > opt[f"hi{st + 1}"] + 0.05:
                opt[f"hi{st + 1}"] = opt[f"hi{st}"] - 0.05
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
        for st in range(1, nst):
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

        # Finding a matching solution from the already generated DataFrame
        solution = sols[sols == opt].dropna(thresh=len(sols.columns) - 4)
        solution = sols.loc[solution.index]

        if solution.empty:
            raise ValueError("[EXCEPTION] No solution satisfying the period range condition was found!")
        else:
            ipbsd = Master(self.dir)
            solution, modes = ipbsd.get_all_section_combinations(t_lower=None, t_upper=None, solution=solution.iloc[0],
                                                                 data=data)
            return solution, modes

    def cacheRCMRF(self, ipbsd, ipbsd_outputs, case_directory, details, sol, demands):
        """
        Creates cache to be used by RCMRF
        :param ipbsd: object                            Master class
        :param ipbsd_outputs: dict                      IPBSD outputs
        :param case_directory: str                      Directory of the case
        :param details: dict                            Details of design
        :param sol: dict                                Optimal solution
        :param demands: dict                            Demands on structural components
        :return: None
        """
        # Loads
        floorLoad = ipbsd.data.i_d["bldg_ch"][0]
        roofLoad = ipbsd.data.i_d["bldg_ch"][1]
        area = ipbsd.data.i_d["bldg_ch"][2]

        nst = len(ipbsd.data.i_d["h_storeys"])
        nGravity = int(ipbsd.data.i_d["n_gravity_frames"][0])
        nSeismic = int(ipbsd.data.i_d["n_seismic_frames"][0])

        spansX = np.array([ipbsd.data.i_d["spans_X"][x] for x in ipbsd.data.i_d["spans_X"]])
        spansY = np.array([ipbsd.data.i_d["spans_Y"][x] for x in ipbsd.data.i_d["spans_Y"]])
        heights = np.array([ipbsd.data.i_d["h_storeys"][x] for x in ipbsd.data.i_d["h_storeys"]])

        X = sum(spansX)
        Y = sum(spansY)

        if self.system == "Perimeter":
            distLength = spansY[0] / 2
        else:
            distLength = spansY[0]

        # Distributed loads
        distLoads = [floorLoad * distLength, roofLoad * distLength]

        # Point loads, for now will be left as zero
        pLoads = 0.0

        # PDelta loads/ essentially loads going to the gravity frames
        if nGravity > 0:
            distLength = spansY[0]
            pDeltaLoad = [floorLoad * distLength * X, roofLoad * distLength * X]
        else:
            pDeltaLoad = 0.0

        # Masses
        masses = np.array(ipbsd_outputs["lateral loads"]["m"])

        # Creating a DataFrame for loads
        loads = pd.DataFrame(columns=["Storey", "Pattern", "Load"])

        for st in range(1, nst + 1):
            l = distLoads[1] if st == nst else distLoads[0]
            loads = loads.append({"Storey": st,
                                  "Pattern": "distributed",
                                  "Load": l}, ignore_index=True)

            # Point loads will be left as zeros for now
            loads = loads.append({"Storey": st,
                                  "Pattern": "point internal",
                                  "Load": pLoads}, ignore_index=True)
            loads = loads.append({"Storey": st,
                                  "Pattern": "point external",
                                  "Load": pLoads}, ignore_index=True)

            # PDelta loads
            if nGravity > 0:
                l = pDeltaLoad[1] if st == nst else pDeltaLoad[0]
                loads = loads.append({"Storey": st,
                                      "Pattern": "pdelta",
                                      "Load": l}, ignore_index=True)
            else:
                loads = loads.append({"Storey": st,
                                      "Pattern": "pdelta",
                                      "Load": pDeltaLoad}, ignore_index=True)

            # Masses
            loads = loads.append({"Storey": st,
                                  "Pattern": "mass",
                                  "Load": masses[st - 1]}, ignore_index=True)

        # Storing loads
        filepath = case_directory / "loads"
        loads.to_csv(f"{filepath}.csv", index=False)

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

        # Storing the materials file
        filepath = case_directory / "materials"
        materials.to_csv(f"{filepath}.csv", index=False)

        """Section properties"""
        def get_sections(i, sections, details, sol, demands, ele, iterator, st, bay, nbays, bayName):
            # TODO, remove constraint on beams being uniform along the building
            eleType = "Beam" if ele == "Beams" else "Column"
            pos = "External" if bay == 1 else "Internal"
            p0 = pos[0].lower() if eleType == "Column" else ""

            # C-S dimensions
            if eleType == "Beam":
                b = sol.loc[f"b{st}"]
                h = sol.loc[f"h{st}"]
                Ptotal = 0.0
                length = spansX[bay - 1]
                MyPos = details["details"][ele]["Pos"][i][3]["yield"]["moment"]
                MyNeg = details["details"][ele]["Neg"][i][3]["yield"]["moment"]
                coverPos = details["details"][ele]["Pos"][i][0]["cover"]
                coverNeg = details["details"][ele]["Neg"][i][0]["cover"]
                roPos = details["details"][ele]["Pos"][i][0]["reinforcement"] / \
                        (b * (h - coverPos))
                roNeg = details["details"][ele]["Neg"][i][0]["reinforcement"] / \
                        (b * (h - coverNeg))

            else:
                b = h = sol.loc[f"h{p0}{st}"]
                Ptotal = max(demands[ele]["N"][st - 1][bay - 1], demands[ele]["N"][st - 1][nbays - bay + 1],
                             key=abs)
                length = heights[st - 1]
                MyPos = MyNeg = details["details"][ele][i][3]["yield"]["moment"]
                coverPos = coverNeg = details["details"][ele][i][0]["cover"]
                roPos = roNeg = details["details"][ele][i][0]["reinforcement"] / (b * (h - coverPos))

            # Residual strength of component
            res = iterator[i][3]["fracturing"]["moment"] / iterator[i][3]["yield"]["moment"]

            # TODO, record beam + and - moments in IPBSD, also for ro and db
            # Appending into the DataFrame
            sections = sections.append({"Element": eleType,
                                        "Bay": bayName,
                                        "Storey": st,
                                        "Position": pos,
                                        "b": float(b),
                                        "h": float(h),
                                        "coverPos": float(coverPos),
                                        "coverNeg": float(coverNeg),
                                        "Ptotal": Ptotal,
                                        "MyPos": float(MyPos),
                                        "MyNeg": float(MyNeg),
                                        "asl": asl,
                                        "Ash": float(iterator[i][0]["A_sh"]),
                                        "spacing": float(iterator[i][0]["spacing"]),
                                        "db": db,
                                        "c": c,
                                        "D": D,
                                        "Res": float(res),
                                        "Length": length,
                                        "ro_long_pos": float(roPos),
                                        "ro_long_neg": float(roNeg)}, ignore_index=True)
            return sections

        # Constants - assumptions - to be made more flexible
        asl = 0
        c = 1
        D = 1
        db = 20
        nbays = len(spansY)

        # Initialize sections
        sections = pd.DataFrame(columns=["Element", "Bay", "Storey", "Position", "b", "h", "coverPos", "coverNeg",
                                         "Ptotal", "MyPos", "MyNeg", "asl", "Ash", "spacing", "db", "c", "D", "Res",
                                         "Length", "ro_long_pos", "ro_long_neg"])

        for ele in details["details"]:
            if ele == "Beams":
                iterator = details["details"][ele]["Pos"]
            else:
                iterator = details["details"][ele]

            for i in iterator:
                st = int(i[1])
                bay = int(i[3])
                sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bay, nbays, bayName=bay)

        # Add symmetric columns
        if bay < nbays:
            bayName = bay
            for bb in range(nbays - bay + 1, 0, -1):
                bayName += 1
                for st in range(1, nst + 1):
                    ele = "Columns"
                    i = f"S{st}B{bb}"
                    iterator = details["details"][ele]
                    sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bb, nbays, bayName)

            bayName = bay
            for bb in range(nbays - bay, 0, -1):
                bayName += 1
                for st in range(1, nst + 1):
                    ele = "Beams"
                    i = f"S{st}B{bb}"
                    iterator = details["details"][ele]["Pos"]
                    sections = get_sections(i, sections, details, sol, demands, ele, iterator, st, bb, nbays, bayName)

        # Storing the materials file
        filepath = case_directory / "sections"
        sections.to_csv(f"{filepath}.csv", index=False)

    def run_ipbsd(self):

        """Calling the master file"""
        ipbsd = Master(self.dir)

        """Generating and storing the input arguments"""
        ipbsd.read_input(self.input_file, self.hazard_file, self.outputPath)
        # Initiate {project name}
        print(f"[INITIATE] Starting IPBSD for {ipbsd.data.case_id}")

        print("[PHASE] Commencing phase 1...")
        self.create_folder(self.outputPath)
        ipbsd.data.i_d["MAFC"] = self.target_MAFC
        ipbsd.data.i_d["EAL"] = self.limit_EAL
        # Store IPBSD inputs as a json
        if self.record:
            self.store_results(self.outputPath / "input", ipbsd.data.i_d, "json")
        print("[SUCCESS] Input arguments have been read and successfully stored")

        # """Get EAL"""
        lam_ls = ipbsd.get_hazard_pga(self.target_MAFC)
        eal = ipbsd.get_loss_curve(lam_ls, self.limit_EAL)

        """Get design limits"""
        theta_max, a_max = ipbsd.get_design_values(self.slfDir, self.geometry)
        print("[SUCCESS] SLF successfully read, and design limits are calculated")

        # """Transform design values into spectral coordinates"""
        # table_sls, delta_spectral, alpha_spectral = ipbsd.perform_transformations(theta_max, a_max)
        # print("[SUCCESS] Spectral values of design limits are obtained")

        # """Get spectra at SLS"""
        # print("[PHASE] Commencing phase 2...")
        # sa, sd, period_range = ipbsd.get_spectra(lam_ls[1])
        # print("[SUCCESS] Response spectra generated")
        #
        # """Get feasible fundamental period range"""
        # t_lower, t_upper = ipbsd.get_period_range(delta_spectral, alpha_spectral, sd, sa)
        # print("[SUCCESS] Feasible period range identified")
        #
        # """Get all section combinations satisfying period bound range"""
        # sols, opt_sol, opt_modes = ipbsd.get_all_section_combinations(t_lower, t_upper, fstiff=self.fstiff)
        # print("[SUCCESS] All section combinations were identified")
        #
        # """Perform SPO2IDA"""
        # print("[PHASE] Commencing phase 3...")
        # Based on available literature, depending on perimeter or space frame, inclusion of gravity loads
        # if ipbsd.data.n_gravity > 0:
        #     if self.analysis_type == 3 or self.analysis_type == 5:
        #         overstrength = 1.3
        #     else:
        #         overstrength = 1.0
        # else:
        #     overstrength = 2.5
        #
        # part_factor, m_star, say, dy = self.iterate_phase_3(ipbsd, case_directory, opt_sol, overstrength)
        #
        # """Get action and demands"""
        # print("[PHASE] Commencing phase 4...")
        # phase_4 = self.iterate_phase_4(ipbsd, say, dy, sa, period_range, opt_sol,
        #                                opt_modes, table_sls, t_lower, t_upper)
        # forces = next(phase_4)
        # demands = next(phase_4)
        #
        # details, hard_ductility, fract_ductility = next(phase_4)
        # warn, warnings = next(phase_4)

        # if self.iterate and warn == 1:
        #     print("[ITERATION 4a] Commencing iteration...")
        #     cnt = 1
        #     while warn == 1 and cnt <= self.maxiter + 1:
        #         """Look for a different solution"""
        #         print(f"[ITERATION 4a] Iteration: {cnt}")
        #         opt_sol, opt_modes = self.seek_solution(ipbsd.data, warnings, sols)
        #
        #         # Redo phases 3 and 4
        #         part_factor, m_star, say, dy = self.iterate_phase_3(ipbsd, case_directory, opt_sol, overstrength)
        #         phase_4 = self.iterate_phase_4(ipbsd, say, dy, sa, period_range, opt_sol,
        #                                        opt_modes, table_sls, t_lower, t_upper)
        #         forces = next(phase_4)
        #         demands = next(phase_4)
        #         details, hard_ductility, fract_ductility = next(phase_4)
        #         warn = next(phase_4)
        #         cnt += 1
        #
        # """Storing the outputs"""
        # # Storing the IPBSD outputs
        # if self.record:
        #     self.store_results(case_directory/"section_combos", sols, "csv")
        #     self.store_results(case_directory/"optimal_solution", opt_sol, "csv")
        #     ipbsd_outputs = {"MAFC": self.target_MAFC, "EAL": eal, "theta_max": theta_max, "alpha_max": alpha_max,
        #                      "part_factor": part_factor, "Mstar": m_star, "delta_spectral": delta_spectral,
        #                      "alpha_spectral": alpha_spectral, "Period range": [t_lower, t_upper],
        #                      "overstrength": overstrength, "yield": [say, dy], "lateral loads": forces}
        #     self.store_results(case_directory / "demands", demands, "pkl")
        #     self.store_results(case_directory / "ipbsd", ipbsd_outputs, "pkl")
        #     design_results = {"details": details, "hardening ductility": hard_ductility,
        #                       "fracturing ductility": fract_ductility}
        #     self.store_results(case_directory / "details", design_results, "pkl")
        #
        #     """Creating DataFrames to store for RCMRF input"""
        #     self.cacheRCMRF(ipbsd, ipbsd_outputs, case_directory, design_results, opt_sol, demands)
        #
        # print("[SUCCESS] Structural elements were designed and detailed. SPO curve parameters were estimated")

        # TODO, add iteration for SPO, I don't think it is yet added
        # """Perform eigenvalue analysis on designed frame"""
        # print("[PHASE] Commencing phase 5...")
        # # Estimates period which is based on calculated EI values from M-phi relationships. Not a mandatory step.
        # # the closer 'period_stf' to 'period' the better the accuracy of the assumption of 50% inertia reduction
        # period_stf, phi = ipbsd.run_ma(opt_sol, t_lower, t_upper, details)
        #
        # # Note: stiffness based off first yield point, at nominal point the stiffness is actually lower, and might act
        # # as a more realistic value. Notably Haselton, 2016 limits the secant yield stiffness between 0.2EIg and 0.6EIg.
        # ipbsd.verify_period(round(period_stf, 2), t_lower, t_upper)
        # print("[SUCCESS] Fundamental period has been verified.")

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
    from pathlib import Path
    path = Path.cwd()
    outputPath = path.parents[0] / ".applications/case1/Output"

    # Add input arguments
    analysis_type = 3
    input_file = path.parents[0] / ".applications/case1/ipbsd_input.csv"
    hazard_file = path.parents[0] / ".applications/case1/Hazard-LAquila-Soil-C.pkl"
    slfDir = path.parents[0] / ".applications/case1/Output/slfoutput"
    spo_file = path.parents[0] / ".applications/case1/spo.csv"
    limit_eal = 0.8
    mafc_target = 2.e-4
    damping = .05
    system = "Perimeter"
    maxiter = 20
    fstiff = 0.5
    geometry = "2d"

    method = IPBSD(input_file, hazard_file, slfDir, spo_file, limit_eal, mafc_target, outputPath, analysis_type,
                   damping=damping, num_modes=2, record=True, iterate=True, system=system, maxiter=20, fstiff=0.5,
                   geometry=geometry)
    start_time = method.get_init_time()
    method.run_ipbsd()
    method.get_time(start_time)
