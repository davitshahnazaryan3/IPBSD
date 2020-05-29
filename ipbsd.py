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
    def __init__(self, input_file, hazard_file, slf_file, spo_file, limit_eal, target_mafc, analysis_type=1,
                 damping=.05, num_modes=3, record=True, iterate=False):
        """
        Initializes IPBSD
        :param input_file: str              Input filename as '*.csv'
        :param hazard_file: str             Hazard filename as '*.pkl' or '*.pickle'
        :param slf_file: str                SLF filename '*.xlsx'
        :param spo_file: str                SPO filename '*.csv'
        :param limit_eal: float             Liming value of EAL
        :param target_mafc: float           Target value of MAFC
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
        """
        self.dir = Path.cwd()
        self.input_file = input_file
        self.hazard_file = hazard_file
        self.slf_file = slf_file
        self.spo_file = spo_file
        self.limit_EAL = limit_eal
        self.target_MAFC = target_mafc
        self.analysis_type = analysis_type
        self.num_modes = num_modes
        self.damping = damping
        self.record = record
        self.iterate = iterate

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
        :param t_upper: flaot                           Upper period limit
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
            demands = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]))
        elif self.analysis_type == 3:
            demands = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]))
        elif self.analysis_type == 4 or self.analysis_type == 5:
            demands = {}
            for mode in range(self.num_modes):
                demands[f"Mode{mode+1}"] = ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"][mode]))
            demands = ipbsd.perform_cqc(corr, demands)
            if self.analysis_type == 5:
                demands_gravity = ipbsd.run_analysis(self.analysis_type, solution, grav_loads=list(forces["G"]))
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
                                                                                         t_lower, t_upper, dy)
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

    def run_ipbsd(self):

        """Calling the master file"""
        ipbsd = Master(self.dir)

        """Generating and storing the input arguments"""
        ipbsd.read_input(self.input_file, self.hazard_file)
        database_directory = self.dir/"Database"
        case_directory = database_directory/ipbsd.data.case_id
        print(f"[INITIATE] Starting IPBSD for {ipbsd.data.case_id}")

        print("[PHASE] Commencing phase 1...")
        self.create_folder(case_directory)
        ipbsd.data.i_d["MAFC"] = self.target_MAFC
        ipbsd.data.i_d["EAL"] = self.limit_EAL
        if self.record:
            self.store_results(case_directory / "input", ipbsd.data.i_d, "json")
        print("[SUCCESS] Input arguments have been read and successfully stored")

        """Get EAL"""
        lam_ls = ipbsd.get_hazard_pga(self.target_MAFC)
        eal = ipbsd.get_loss_curve(lam_ls, self.limit_EAL)

        """Get design limits"""
        theta_max, alpha_max = ipbsd.get_design_values(self.slf_file)
        print("[SUCCESS] SLF successfully read, and design limits are calculated")

        """Transform design values into spectral coordinates"""
        table_sls, delta_spectral, alpha_spectral = ipbsd.perform_transformations(theta_max, alpha_max)
        print("[SUCCESS] Spectral values of design limits are obtained")

        """Get spectra at SLS"""
        print("[PHASE] Commencing phase 2...")
        sa, sd, period_range = ipbsd.get_spectra(lam_ls[1])
        print("[SUCCESS] Response spectra generated")

        """Get feasible fundamental period range"""
        t_lower, t_upper = ipbsd.get_period_range(delta_spectral, alpha_spectral, sd, sa)
        print("[SUCCESS] Feasible period range identified")

        """Get all section combinations satisfying period bound range"""
        sols, opt_sol, opt_modes = ipbsd.get_all_section_combinations(t_lower, t_upper)
        print("[SUCCESS] All section combinations were identified")

        """Perform SPO2IDA"""
        print("[PHASE] Commencing phase 3...")
        # Based on available literature, depending on perimeter or space frame, inclusion of gravity loads
        if ipbsd.data.n_gravity > 0:
            if self.analysis_type == 3 or self.analysis_type == 5:
                overstrength = 1.3
            else:
                overstrength = 1.0
        else:
            overstrength = 2.5

        part_factor, m_star, say, dy = self.iterate_phase_3(ipbsd, case_directory, opt_sol, overstrength)

        """Get action and demands"""
        print("[PHASE] Commencing phase 4...")
        phase_4 = self.iterate_phase_4(ipbsd, say, dy, sa, period_range, opt_sol,
                                       opt_modes, table_sls, t_lower, t_upper)
        forces = next(phase_4)
        demands = next(phase_4)
        details, hard_ductility, fract_ductility = next(phase_4)
        warn, warnings = next(phase_4)

        if self.iterate and warn == 1:
            print("[ITERATION 4a] Commencing iteration...")
            cnt = 1
            while warn == 1:
                """Look for a different solution"""
                print(f"[ITERATION 4a] Iteration: {cnt}")
                opt_sol, opt_modes = self.seek_solution(ipbsd.data, warnings, sols)

                # Redo phases 3 and 4
                part_factor, m_star, say, dy = self.iterate_phase_3(ipbsd, case_directory, opt_sol, overstrength)
                phase_4 = self.iterate_phase_4(ipbsd, say, dy, sa, period_range, opt_sol,
                                               opt_modes, table_sls, t_lower, t_upper)
                forces = next(phase_4)
                demands = next(phase_4)
                details, hard_ductility, fract_ductility = next(phase_4)
                warn = next(phase_4)
                cnt += 1

        """Storing the outputs"""
        # Storing the IPBSD outputs
        if self.record:
            self.store_results(case_directory/"section_combos", sols, "csv")
            self.store_results(case_directory/"optimal_solution", opt_sol, "csv")
            ipbsd_outputs = {"MAFC": self.target_MAFC, "EAL": eal, "theta_max": theta_max, "alpha_max": alpha_max,
                             "part_factor": part_factor, "Mstar": m_star, "delta_spectral": delta_spectral,
                             "alpha_spectral": alpha_spectral, "Period range": [t_lower, t_upper],
                             "overstrength": overstrength, "yield": [say, dy], "lateral loads": forces}
            self.store_results(case_directory / "demands", demands, "pkl")
            self.store_results(case_directory / "ipbsd", ipbsd_outputs, "pkl")
            design_results = {"details": details, "hardening ductility": hard_ductility,
                              "fracturing ductility": fract_ductility}
            self.store_results(case_directory / "details", design_results, "pkl")
        print("[SUCCESS] Structural elements were designed and detailed. SPO curve parameters were estimated")

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
    """
    # Add input arguments
    analysis_type = 2
    input_file = "input.csv"
    hazard_file = "Hazard-LAquila-Soil-C.pkl"
    slf_file = "slf.xlsx"
    spo_file = "spo.csv"
    limit_eal = 0.5
    mafc_target = 1.e-4
    damping = .05

    method = IPBSD(input_file, hazard_file, slf_file, spo_file, limit_eal, mafc_target, analysis_type,
                   damping=damping, num_modes=2, record=True, iterate=True)
    start_time = method.get_init_time()
    method.run_ipbsd()
    method.get_time(start_time)
