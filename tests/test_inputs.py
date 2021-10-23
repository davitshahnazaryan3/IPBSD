import unittest
import os
from pathlib import Path

from src.input import Input


class TestInputs(unittest.TestCase):
    path = Path.cwd()
    outputPath = path.parents[0] / "sample/sample1"

    # Set FLAG for 3D here
    flag3d = True
    # Set FILENAME here
    FILENAME = outputPath / "ipbsd_input.csv"

    def test_performance_limit_states(self):
        """
        Verify limit state names, expected loss ratios, return periods, aleatory uncertainties
        """
        data = Input(self.flag3d)
        data.read_inputs(self.FILENAME)

        ents = data._get_performance_limit_states()

        # type checks
        self.assertIsInstance(ents["PLS"][0], str, "PLS entities must be strings!")
        self.assertIsInstance(ents["y"][0], float, "y entities must be numbers!")
        self.assertIsInstance(ents["TR"][0], int, "TR entities must be whole numbers (positive integers)!")
        self.assertIsInstance(ents["beta_al"][0], float, "beta_al entities must be whole numbers (positive integers)!")

        # test ELR, y, values
        self.assertGreater(1, ents["y"][0], "y at 1st LS must be lower than 1!")
        self.assertGreater(1, ents["y"][1], "y at 2nd LS must be lower than 1!")
        self.assertGreater(ents["y"][1], ents["y"][0], "y at 2nd LS must be higher than at 1st LS!")
        self.assertGreaterEqual(ents["y"][2], 1, "y at 3rd LS must be higher or equal than 1!")

    def test_loads(self):
        """
        Verify loads supplied are correct
        """
        data = Input(self.flag3d)
        data.read_inputs(self.FILENAME)

        loads = data._get_loads()

        # size
        self.assertEqual(len(loads), 2, "You must supply load values for permanent and live loads!")

        # type checks
        for i in loads:
            self.assertIsInstance(loads[i], float, "Load values must be numbers!")

    def test_mode_reduction_factor(self):
        data = Input(self.flag3d)
        data.read_inputs(self.FILENAME)
        factor = data._get_mode_reduction_factor()

        self.assertIsInstance(factor, float, "Mode reduction factor must be a number!")

    def test_building_dimensions(self):
        data = Input(self.flag3d)
        data.read_inputs(self.FILENAME)

        h, x, y, config = data._get_building_dimensions()

        self.assertIsInstance(config, str, "System configuration must be a text (string)!")
        if self.flag3d:
            self.assertEqual(config, "space", "System configuration must be space for 3D modelling!")
        self.assertTrue(config.lower() == "space" or config.lower() == "perimeter",
                        "System configuration must be Space or Perimeter")

        for i in h:
            self.assertIsInstance(i, float, "Building storey heights must be numbers!")
            if i > 5:
                print(f"----[WARNING], Storey height of {i} is larger than 5 metres!")

        for i in x:
            self.assertIsInstance(i, float, "Building bays (spans_X) in X direction must be numbers")
            if i > 8:
                print(f"----[WARNING], Bay width in X of {i} is larger than 8 metres!")

        for i in y:
            self.assertIsInstance(i, float, "Building bays (spans_Y) in Y direction must be numbers")
            if i > 8:
                print(f"----[WARNING], Bay width in Y of {i} is larger than 8 metres!")

    def test_material_properties(self):
        data = Input(self.flag3d)
        data.read_inputs(self.FILENAME)

        ents = data._get_material_props()
        for i in ents:
            self.assertIsInstance(i, float, "Material properties must be supplied as numbers!")


def assertIsFile(path, msg):
    if not path.resolve().is_file():
        raise AssertionError(msg)


def assertIsDir(path, msg):
    if not path.resolve().is_dir():
        raise AssertionError(msg)


class TestMain(unittest.TestCase):
    # Set input parameters here --->
    path = Path.cwd()
    outputPath = path.parents[0] / "sample/sample1"

    # Set FLAG for 3D here
    flag3d = True

    # Directories and path to files
    input_filename = outputPath / "ipbsd_input.csv"
    hazard_filename = outputPath / "hazard/hazard.pkl"
    spo_filename = outputPath / "spo.csv"
    slf_directory = outputPath / "slfoutput"

    limit_eal = 1.0
    mafc_target = 2.e-4
    analysis_type = 3
    damping = .05
    num_modes = 2
    fstiff = 0.5
    rebar_cover = 0.03
    overstrength = 1.0
    repl_cost = 349459.2
    iterate = True
    export = True
    hold_flag = False
    eal_correction = True
    perform_scaling = True
    maxiter = 10
    gravity_cs = None
    solutionFileX = None
    solutionFileY = None
    solutionFile = None

    def test_input_path_validity(self):
        assertIsFile(self.input_filename, "Input file path incorrect!")

    def test_hazard_path_validity(self):
        assertIsFile(self.hazard_filename, "Hazard file path incorrect!")

    def test_spo_path_validity(self):
        assertIsFile(self.spo_filename, "SPO file path incorrect!")

    def test_slf_directory_or_path_validity(self):
        self.assertTrue(self.slf_directory.resolve().is_file() or self.slf_directory.resolve().is_dir(),
                        "SLF directory or file path incorrect!")

    def test_output_directory_validity(self):
        assertIsDir(self.outputPath, "Output directory path incorrect!")

    def test_file_types(self):
        self.assertEqual(str(self.input_filename)[-3:], "csv", "Input must be a .csv file!")
        self.assertTrue(str(self.hazard_filename).endswith("pkl") or str(self.hazard_filename).endswith("pickle"),
                        "Hazard should be pkl or pickle file containing a dictionary object!")
        self.assertEqual(str(self.spo_filename)[-3:], "csv", "SPO must be a .csv file!")

    def test_file_validity_in_slf_directory(self):
        if os.path.isdir(self.slf_directory):
            # Test only if the variable is a directory
            cnt = 1
            for file in os.listdir(self.slf_directory):
                self.assertTrue(str(file).endswith("csv") or str(file).endswith("pkl") or str(file).endswith("pickle"),
                                "Files in the SLF directory must be either csv or pickle!")
                cnt += 1

                if str(file).endswith("csv"):
                    self.assertGreater(cnt, 1, "There must be only 1 csv file in the SLF directory!")
                    break

    def test_performance_objectives(self):
        self.assertIsInstance(self.mafc_target, float, "MAFC target must be a number!")
        self.assertIsInstance(self.limit_eal, float, "EAL limit must be a number!")

        # Warnings against inconsiderate performance objectives
        if self.mafc_target > 1.e-3:
            print(f"----[WARNING], MAFC target of {self.mafc_target} is too large!")
        if self.mafc_target < 1.e-7:
            print(f"----[WARNING], MAFC target of {self.mafc_target} is too small!")
        if self.limit_eal > 5.:
            print(f"----[WARNING], EAL limit of {self.mafc_target} is too large!")
        if self.limit_eal < 0.1:
            print(f"----[WARNING], EAL limit of {self.mafc_target} is too small!")

    def test_analysis_option(self):
        self.assertTrue(1 <= self.analysis_type <= 5 and isinstance(self.analysis_type, int),
                        "Analysis type must be a number from 1 to 5!")
        self.assertTrue(1 <= self.analysis_type <= 3, "Options 4 and 5 are not yet implemented!")

    def test_damping(self):
        self.assertTrue(0.0 < self.damping < 1.0, "Damping must be in (0, 1) range!")

    def test_num_modes(self):
        self.assertTrue(self.num_modes > 0, "Number of modes must be larger or equal to 1!")
        if self.flag3d:
            self.assertTrue(self.num_modes > 1, "Number of modes must be larger or equal to 2!")

    def test_booleans(self):
        self.assertIsInstance(self.iterate, bool, "iterate must be True or False!")
        self.assertIsInstance(self.flag3d, bool, "flag3d must be True or False!")
        self.assertIsInstance(self.export, bool, "export must be True or False!")
        self.assertIsInstance(self.hold_flag, bool, "hold_flag must be True or False!")
        self.assertIsInstance(self.eal_correction, bool, "eal_correction must be True or False!")
        self.assertIsInstance(self.perform_scaling, bool, "perform_scaling must be True or False!")

    def test_maximum_iterations_variable(self):
        if self.iterate:
            # only when iterating
            self.assertTrue(self.maxiter > 1, "maxiter (maximum number of iterations) must be larger than 1!")

    def test_stiffness_reduction_factor(self):
        self.assertTrue(self.fstiff > 0.0, "fstiff (stiffness reduction factor) must be a number larger than 0!")

    def test_reinforcement_cover(self):
        self.assertTrue(self.rebar_cover <= 0.2, "rebar_cover (reinforcement cover) is too large!")
        self.assertTrue(self.rebar_cover >= 0.01, "rebar_cover (reinforcement cover) is too small!")

    def test_overstrength(self):
        self.assertTrue(self.overstrength >= 1.0, "Overstrength must be larger or equal to 1.0!")

    def test_replacement_cost(self):
        self.assertIsInstance(self.repl_cost, float, "Replacement cost must be a number!")

    def test_solution_files_if_not_none(self):
        def verify(var, file_type, msg):
            assertIsFile(var, msg)
            if file_type == "csv":
                self.assertEqual(str(var)[-3:], file_type, msg)
            else:
                self.assertTrue(str(var).endswith("pkl") or str(var).endswith("pickle"), msg)

        if self.gravity_cs:
            verify(self.gravity_cs, "csv", "Path to gravity solutions is incorrect! Must be a csv file!")

        if self.solutionFile:
            if not isinstance(self.solutionFile, int):
                verify(self.solutionFile, "pkl", "Path to solutions file is incorrect! Must be pkl or pickle")

        if self.solutionFileX:
            if not isinstance(self.solutionFileX, int):
                verify(self.solutionFileX, "csv", "Path to solutions file in X direction is incorrect! "
                                                  "Must be a csv file!")

        if self.solutionFileY:
            if not isinstance(self.solutionFileY, int):
                verify(self.solutionFileY, "csv", "Path to solutions file in Y direction is incorrect! "
                                                  "Must be a csv file!")
