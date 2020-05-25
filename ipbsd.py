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
    def __init__(self, input_file, hazard_file, slf_file, spo_file, limit_eal, target_mafc, mc_my, analysis_type=1,
                 damping=.05, num_modes=3):
        """
        Initializes IPBSD
        :param input_file: str              Input filename as '*.csv'
        :param hazard_file: str             Hazard filename as '*.pkl' or '*.pickle'
        :param slf_file: str                SLF filename '*.xlsx'
        :param spo_file: str                SPO filename '*.csv'
        :param limit_eal: float             Liming value of EAL
        :param target_mafc: float           Target value of MAFC
        :param mc_my: float                 Peak to yield ratio
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
        """
        self.dir = Path.cwd()
        self.input_file = input_file
        self.hazard_file = hazard_file
        self.slf_file = slf_file
        self.spo_file = spo_file
        self.limit_EAL = limit_eal
        self.target_MAFC = target_mafc
        self.mc_my = mc_my
        self.analysis_type = analysis_type
        self.num_modes = num_modes
        self.damping = damping

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
        self.store_results(case_directory/"section_combos", sols, "csv")
        self.store_results(case_directory/"optimal_solution", opt_sol, "csv")
        print("[SUCCESS] All section combinations were identified")

        """Perform SPO2IDA"""
        print("[PHASE] Commencing phase 3...")
        period = round(float(opt_sol['T']), 1)
        spo_data = ipbsd.data.initial_spo_data(period, self.dir/"client"/self.spo_file)
        spo2ida_data = ipbsd.perform_spo2ida(spo_data)
        self.store_results(case_directory/"spo2ida_results", spo2ida_data, "pkl")
        print("[SUCCESS] SPO2IDA was performed")

        """Yield strength optimization for MAFC and verification"""
        # Based on available literature, depending on perimeter or space frame, inclusion of gravity loads
        if ipbsd.data.n_gravity > 0:
            if self.analysis_type == 3 or self.analysis_type == 5:
                overstrength = 1.6
            else:
                overstrength = 1.3
        else:
            overstrength = 2.5

        part_factor = opt_sol["Part Factor"]
        say, dy = ipbsd.verify_mafc(period, spo2ida_data, part_factor, self.target_MAFC, overstrength, hazard="True")
        print("[SUCCESS] MAFC was validated")

        """Get action and demands"""
        print("[PHASE] Commencing phase 4...")
        if self.analysis_type == 4 or self.analysis_type == 5:
            se_rmsa = ipbsd.get_sa_at_period(say, sa, period_range, opt_modes["Periods"])
        else:
            se_rmsa = None

        self.num_modes = min(self.num_modes, ipbsd.data.nst)
        if self.analysis_type == 4 or self.analysis_type == 5:
            corr = ipbsd.get_correlation_matrix(opt_modes["Periods"], self.num_modes, damping=self.damping)
        else:
            corr = None
        forces = ipbsd.get_action(opt_sol, say, pd.DataFrame.from_dict(table_sls),
                                  ipbsd.data.w_seismic, self.analysis_type, self.num_modes, opt_modes, modal_sa=se_rmsa)
        print("[SUCCESS] Actions on the structure for analysis were estimated")

        """Perform ELF analysis"""
        if self.analysis_type == 1:
            demands = ipbsd.run_muto_approach(opt_sol, list(forces["Fi"]), ipbsd.data.h, ipbsd.data.spans_x)
        elif self.analysis_type == 2:
            demands = ipbsd.run_analysis(self.analysis_type, opt_sol, list(forces["Fi"]))
        elif self.analysis_type == 3:
            demands = ipbsd.run_analysis(self.analysis_type, opt_sol, list(forces["Fi"]), list(forces["G"]))
        elif self.analysis_type == 4 or self.analysis_type == 5:
            demands = {}
            for mode in range(self.num_modes):
                demands[f"Mode{mode+1}"] = ipbsd.run_analysis(self.analysis_type, opt_sol, list(forces["Fi"][mode]))
            demands = ipbsd.perform_cqc(corr, demands)
            if self.analysis_type == 5:
                demands_gravity = ipbsd.run_analysis(self.analysis_type, opt_sol, grav_loads=list(forces["G"]))
                # Combining gravity and RSMA results
                for eleType in demands_gravity.keys():
                    for dem in demands_gravity[eleType].keys():
                        demands[eleType][dem] = demands[eleType][dem] + demands_gravity[eleType][dem]

        else:
            raise ValueError("[EXCEPTION] Incorrect analysis type...")

        self.store_results(case_directory/"demands", demands, "pkl")
        print("[SUCCESS] Analysis completed and demands on structural elements were estimated.")

        # Storing the IPBSD outputs
        ipbsd_outputs = {"MAFC": lam_ls, "EAL": eal, "theta_max": theta_max, "alpha_max": alpha_max,
                         "part_factor": part_factor, "delta_spectral": delta_spectral, "alpha_spectral": alpha_spectral,
                         "Period range": [t_lower, t_upper], "overstrength": overstrength, "yield": [say, dy],
                         "lateral loads": forces}
        self.store_results(case_directory/"ipbsd", ipbsd_outputs, "pkl")

        # todo, store details necessary for constructing the model in OpenSees
        """Design the structural elements"""
        details, hard_ductility = ipbsd.design_elements(demands, opt_sol, t_lower, t_upper, dy)
        """Estimate parameters for SPO curve and compare with assumed shape"""
        design_results = {"details": details, "hardening ductility": hard_ductility}
        self.store_results(case_directory / "details", design_results, "pkl")
        print("[SUCCESS] Structural elements were designed and detailed. SPO curve parameters were estimated")

        """Perform eigenvalue analysis on designed frame"""
        print("[PHASE] Commencing phase 5...")
        # Estimates period which is based on calculated EI values from M-phi relationships. Not a mandatory step.
        # the closer 'period_stf' to 'period' the better the accuracy of the assumption of 50% inertia reduction
        period_stf, phi = ipbsd.run_ma(opt_sol, t_lower, t_upper, details)

        # Note: stiffness based off first yield point, at nominal point the stiffness is actually lower, and might act
        # as a more realistic value. Notably Haselton, 2016 limits the secant yield stiffness between 0.2EIg and 0.6EIg.
        ipbsd.verify_period(round(period_stf, 2), t_lower, t_upper)
        print("[SUCCESS] Fundamental period has been verified.")

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
    mc_my = 1.13                                # Following Haselton, 2016

    method = IPBSD(input_file, hazard_file, slf_file, spo_file, limit_eal, mafc_target, mc_my, analysis_type,
                   damping=damping, num_modes=2)
    start_time = method.get_init_time()
    method.run_ipbsd()
    method.get_time(start_time)
