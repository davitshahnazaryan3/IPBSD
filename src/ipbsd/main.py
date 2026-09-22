"""
Runs the integrated seismic risk and economic loss driven framework
"""
from pathlib import Path

from src.master import Master


class Main:
    def __init__(self, input_filename, hazard_filename, spo_filename, slf_directory, limit_eal, target_mafc,
                 output_path, analysis_type=1, damping=.05, num_modes=3, iterate=False, maxiter=20, fstiff=0.5,
                 rebar_cover=0.03, export=False, hold_flag=False, overstrength=None, repl_cost=None,
                 gravity_cs=None, eal_correction=True, perform_scaling=True, solution_filex=None, solution_filey=None,
                 solution_file=None, edp_profiles=None, flag3d=False):
        """
        Initializes IPBSD
        Files:
        :param input_filename: str          Input filename as '*.csv'
        :param hazard_filename: str         Hazard filename as '*.pkl' or '*.pickle'
        :param spo_filename: str            SPO filename '*.csv'
        :param slf_directory: str           Directory of SLFs derived via SLF Generator
        Performance objectives:
        :param limit_eal: float             Liming value of EAL
        :param target_mafc: float           Target value of MAFC
        Directory to store output:
        :param output_path: str             Outputs path (where to store the outputs)
        Optional arguments (default):
        :param analysis_type: int           Analysis type:
                                            1: Simplified ELF - no analysis is run, tools based on
                                            simplified expressions, actions based on 1st mode shape (default)
                                            2: ELF - equivalent lateral force method of analysis, actions based on
                                            1st mode shape
                                            3: ELF & gravity - analysis under ELF and gravity loads
                                            4: RMSA - response method of spectral analysis, actions based on "n" modal
                                            shapes
                                            5: RMSA & gravity - analysis under RMSA and gravity loads
        :param damping: float               Ratio of critical damping
        :param num_modes: int               Number of modes to consider for SRSS (for analysis type 4 and 5)
        :param iterate: bool                Perform iterations or not
        :param maxiter: int                 Maximum number of iterations for seeking a solution
        :param fstiff: float                Stiffness reduction factor
        :param rebar_cover: float           Reinforcement cover in m
        :param export: bool                 Whether to export cache at each major step into an outputs directory or not
        :param hold_flag: bool              Flag to stop the framework once the solution combinations have been computed
                                            This allows the user to subdivide the framework into two sections if it
                                            takes too long to find a solution. Alternatively, if holdFlag is False,
                                            full framework will be run, and if solution.csv at the outputDir already
                                            exists, then the framework will skip the step of its derivation.
                                            holdFlag=False makes sense if the framework has already generated the csv
                                            file containing all valid solutions.
        :param overstrength: float          Overstrength ratio, leave on default and the user will automatically assign
                                            a value
        :param repl_cost: float             Replacement cost of the entire building
        :param gravity_cs: str              Path to gravity solution (for 3D modelling)
        :param eal_correction: bool         Perform EAL correction
        :param perform_scaling: bool        Perform scaling of SLFs to replCost (the scaling should not matter)
        Solution files:
        :param solution_filex: str          Path to solution file to be used for design in X direction
        :param solution_filey: str          Path to solution file to be used for design in Y direction (for 3D)
        :param edp_profiles: list           EDP profile shape to use as a guess
        :param solution_file: str           Solution file containing a dictionary for the Space System, 3D (*.pickle)
        """
        self.input_filename = input_filename
        self.hazard_filename = hazard_filename
        self.spo_filename = spo_filename
        self.slf_directory = slf_directory
        self.limit_eal = limit_eal
        self.target_mafc = target_mafc
        self.output_path = output_path
        self.analysis_type = analysis_type
        self.num_modes = num_modes
        self.damping = damping
        self.iterate = iterate
        self.maxiter = maxiter
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.export = export
        self.hold_flag = hold_flag
        self.overstrength = overstrength
        self.repl_cost = repl_cost
        self.gravity_cs = gravity_cs
        self.eal_correction = eal_correction
        self.perform_scaling = perform_scaling
        self.solution_filex = solution_filex
        self.solution_filey = solution_filey
        self.solution_file = solution_file
        self.edp_profiles = edp_profiles
        self.flag3d = flag3d

    def run_master(self):
        master = Master(self)

        # read inputs
        master.read_input()

        # perform IPBSD calculations
        master.perform_calculations()

        # get all section combinations
        master.get_all_section_combinations()

        # Iterative phase
        if not self.hold_flag:
            master.perform_iterations()


if __name__ == "__main__":
    """
    :param analysis_type: int                   Analysis type:
                                                1: Simplified ELF - no analysis is run, tools based on 
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
    outputPath = path / "sample/sample1"

    # Add input arguments
    analysis_type = 3
    input_file = outputPath / "ipbsd_input.csv"
    hazard_file = outputPath / "hazard/hazard.pkl"
    slfDir = outputPath / "slfoutput"
    spo_file = outputPath / "spo.csv"
    limit_eal = 1.0
    mafc_target = 2.e-4
    damping = .05
    maxiter = 10
    fstiff = 0.5
    overstrength = 1.0
    replCost = 349459.2
    export_cache = True
    holdFlag = True
    iterate = True
    flag3d = True

    gravity_cs = None
    solutionFileX = None
    solutionFileY = None

    # Design solution to use (leave None, if the tool needs to look for the solution)
    # solutionFile = path.parents[0] / ".applications/case1/designSol.csv"
    solutionFile = None

    main = Main(input_file, hazard_file, spo_file, slfDir, limit_eal, mafc_target, outputPath, analysis_type, damping,
                iterate=iterate, maxiter=maxiter, fstiff=fstiff, flag3d=flag3d, export=export_cache,
                overstrength=overstrength, repl_cost=replCost, gravity_cs=gravity_cs, hold_flag=holdFlag)
    main.run_master()
