"""
Optimizes for the fundamental period by seeking cross-sections of all structural elements
"""
from external.openseesrun3d import OpenSeesRun3D
import numpy as np
import constraint
import pandas as pd
import sys


class CrossSectionSpace:
    def __init__(self, ipbsd, period_limits, fstiff, iteration=False, cache_dir=None, reduce_combos=True):
        """
        Initialize
        :param ipbsd: object                        IPBSD input data
        :param period_limits: dict                  Period limits identified via IPBSD
        :param fstiff: float                        Stiffness reduction factor (initial assumption)
        :param iteration: bool                      Whether iterations are being carried out via IPBSD
        :param cache_dir: str                       Directory where cache of solutions needs to be exported to
        :param reduce_combos: bool                  Reduce number of combinations to be created (adds more constraints)
        """
        self.ipbsd = ipbsd
        self.nst = ipbsd.nst
        self.nbays_x = len(ipbsd.spans_x)
        self.nbays_y = len(ipbsd.spans_y)
        self.period_limits = period_limits
        self.fstiff = fstiff
        self.reduce_combos = reduce_combos
        self.SWEIGHT = 25.

        # Export solution as cache in .csv (initialize)
        if cache_dir is not None:
            cache_path = cache_dir
            if not iteration and not cache_path.exists():
                elements_cache_path = cache_path.parents[0] / "elements_space.csv"
                if not elements_cache_path.exists():
                    self.elements = self.constraint_function()
                    # Export solutions as cache in .csv
                    self.elements.to_csv(elements_cache_path)
                else:
                    self.elements = pd.read_csv(elements_cache_path, index_col=[0])

                # Get all solutions within the period limits
                self.solutions = self.get_all_solutions()
                # Export solutions as cache in .csv
                self.solutions.to_csv(cache_path)

            # If solutions file exists, read and derive the solutions
            if cache_path.exists():
                self.solutions = pd.read_csv(cache_path, index_col=[0])

    def get_all_solutions(self):
        """
        Gets all possible solutions respecting the period bounds in both directions
        :return: dict                       All possible solutions within a period range
        """
        hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}
        # Create the DataFrames for the space system
        columns = []
        columns_gr = []
        for st in range(1, self.ipbsd.nst + 1):
            columns.append(f"he{st}")
            columns.append(f"hi{st}")
            columns.append(f"b{st}")
            columns.append(f"h{st}")
            columns_gr.append(f"hi{st}")
            columns_gr.append(f"bx{st}")
            columns_gr.append(f"hx{st}")
            columns_gr.append(f"by{st}")
            columns_gr.append(f"hy{st}")

        # Initialize
        solution_x = pd.DataFrame(columns=columns)
        solution_y = pd.DataFrame(columns=columns)
        solution_gr = pd.DataFrame(columns=columns_gr)
        # Principal modal periods
        solution_x["T"] = ""
        solution_y["T"] = ""
        # Weight of structural components
        solution_x["Weight"] = ""
        solution_y["Weight"] = ""
        # Effective modal masses
        solution_x["Mstar"] = ""
        solution_y["Mstar"] = ""
        # Modal participation factors
        solution_x["Part Factor"] = ""
        solution_y["Part Factor"] = ""

        # Space systems will be used for 3D modelling only
        solutions = pd.DataFrame(columns=self.elements.columns)
        # Principal modal periods
        solutions["T1"] = ""
        solutions["T2"] = ""
        # Weight of structural components
        solutions["Weight"] = ""
        # Effective modal masses
        solutions["Mstar1"] = ""
        solutions["Mstar2"] = ""
        # Modal participation factors
        solutions["Part Factor1"] = ""
        solutions["Part Factor2"] = ""
        cnt = 0
        for i in self.elements.index:
            # Get element cross-sections of solution i
            ele = self.elements.iloc[i]
            # Generate section properties
            cs = self.get_section(ele)
            # Run modal analysis via OpenSeesPy
            op = OpenSeesRun3D(self.ipbsd, cs, self.fstiff, system="space", hinge=hinge)
            op.create_model()
            op.define_masses()
            num_modes = self.nst if self.nst <= 9 else 9
            periods, modalShape, gamma, mstar = op.ma_analysis(num_modes)
            op.wipe()

            # Verify that both periods are within the period limits
            if self.check_target_t(periods[0], self.period_limits["1"]) and \
                    self.check_target_t(periods[1], self.period_limits["2"]):
                weight = self.get_weight(ele)
                solutions = solutions.append(ele, ignore_index=True)
                solutions["T1"].iloc[cnt] = periods[0]
                solutions["T2"].iloc[cnt] = periods[1]
                solutions["Weight"].iloc[cnt] = weight
                solutions["Mstar1"].iloc[cnt] = mstar[0]
                solutions["Mstar2"].iloc[cnt] = mstar[1]
                solutions["Part Factor1"].iloc[cnt] = gamma[0]
                solutions["Part Factor2"].iloc[cnt] = gamma[1]
                cnt += 1

        return solutions

    def get_section(self, ele):
        """
        Reformat cross-section information for readability by OpenSeesRun3D object
        :param ele: Series                      Element cross-sections
        :return: dict                           Dictionary subdivided into x, y and internal frames
        """

        # Create the Series
        cs_x = {}
        cs_y = {}
        cs_int = {}
        for st in range(1, self.nst + 1):
            # X direction
            cs_x[f"he{st}"] = ele[f"h11{st}"]
            cs_x[f"hi{st}"] = ele[f"h21{st}"]
            cs_x[f"b{st}"] = ele[f"bx11{st}"]
            cs_x[f"h{st}"] = ele[f"hx11{st}"]
            # Y direction
            cs_y[f"he{st}"] = ele[f"h11{st}"]
            cs_y[f"hi{st}"] = ele[f"h12{st}"]
            cs_y[f"b{st}"] = ele[f"by11{st}"]
            cs_y[f"h{st}"] = ele[f"hy11{st}"]
            # Internal
            cs_int[f"hi{st}"] = ele[f"h22{st}"]
            cs_int[f"bx{st}"] = ele[f"bx12{st}"]
            cs_int[f"hx{st}"] = ele[f"hx12{st}"]
            cs_int[f"by{st}"] = ele[f"by21{st}"]
            cs_int[f"hy{st}"] = ele[f"hy21{st}"]

        # Into a Series
        cs_x = pd.Series(cs_x, name=ele.name, dtype="float")
        cs_y = pd.Series(cs_y, name=ele.name, dtype="float")
        cs_int = pd.Series(cs_int, name=ele.name, dtype="float")

        # Assign to dictionary (gravity i.e. internal elements)
        cs = {"x_seismic": cs_x, "y_seismic": cs_y, "gravity": cs_int}

        return cs

    def constraint_function(self):
        """
        Constraint functions for identifying the combinations of all possible cross-sections
        We don't want to have a wacky building...
        Unique structural elements: 1. External columns in x and y directions, e.g h111
                                    2. All internal columns, e.g. h221
                                    3. External beams in x direction, e.g. bx111 x hx111
                                    4. External beams in y direction, e.g. by111 x hy111
                                    5. Internal beams in x direction, e.g. bx121 x hx121
                                    6. Internal beams in y direction, e.g. by211 x hy211
        :return: DataFrame                  All solutions with element cross-sections
        """
        # Number of bays
        nx = self.nbays_x
        ny = self.nbays_y

        # Helper constraint functions
        def equality(a, b):
            a = round(a, 2)
            b = round(b, 2)
            if a + 10**-5 >= b >= a - 10**-5:
                return True

        def storey_constraint(a, b):
            a = round(a, 2)
            b = round(b, 2)
            if a + 10**-5 >= b >= a - 0.05 - 10**-5:
                return True

        def bay_constraint(a, b):
            a = round(a, 2)
            b = round(b, 2)
            if self.reduce_combos:
                tol = 0.1
            else:
                tol = 0.2

            if b + tol + 10**-5 >= a >= b - 10**-5:
                return True

        def beam_constraint(a, b):
            a = round(a, 2)
            b = round(b, 2)
            if a + 0.1 - 10**-5 <= b <= a + 0.3 + 10**-5:
                return True

        # Initialize element types
        ele_types = []
        # Initialize the problem
        problem = constraint.Problem()

        # Add the elements into the variables list
        # Loop for each storey level
        for st in range(1, self.nst + 1):
            # Loop for each bay in x direction
            for x in range(1, nx + 2):
                # Loop for each bay in y direction
                for y in range(1, ny + 2):
                    # Add the variables
                    # Columns
                    problem.addVariable(f"h{x}{y}{st}", np.arange(0.35, 0.70, 0.05))
                    ele_types.append(f"h{x}{y}{st}")
                    # Beams along x direction
                    if x < nx + 1:
                        problem.addVariable(f"bx{x}{y}{st}", np.arange(0.35, 0.55, 0.05))
                        problem.addVariable(f"hx{x}{y}{st}", np.arange(0.45, 0.70, 0.05))
                        ele_types.append(f"bx{x}{y}{st}")
                        ele_types.append(f"hx{x}{y}{st}")
                    # Beams along y direction
                    if y < ny + 1:
                        problem.addVariable(f"by{x}{y}{st}", np.arange(0.35, 0.55, 0.05))
                        problem.addVariable(f"hy{x}{y}{st}", np.arange(0.45, 0.70, 0.05))
                        ele_types.append(f"by{x}{y}{st}")
                        ele_types.append(f"hy{x}{y}{st}")

        # Add constraints to cross-section dimensions
        # Constrain symmetry of building
        for st in range(1, self.nst + 1):
            # Constrain columns
            if not self.reduce_combos:
                # Group 1 constraints: Corner columns
                problem.addConstraint(equality, [f"h11{st}", f"h{nx+1}1{st}"])
                problem.addConstraint(equality, [f"h11{st}", f"h{nx+1}{ny+1}{st}"])
                problem.addConstraint(equality, [f"h11{st}", f"h1{ny+1}{st}"])
                # Group 2 constraints: Corner central columns x
                for x in range(2, int(nx / 2) + 2):
                    problem.addConstraint(equality, [f"h{x}1{st}", f"h{x}{ny+1}{st}"])
                    problem.addConstraint(equality, [f"h{x}1{st}", f"h{x+1}1{st}"])
                    problem.addConstraint(equality, [f"h{x}1{st}", f"h{x+1}{ny+1}{st}"])
                # Group 3 constraints: Corner central columns y
                for y in range(2, int(ny / 2) + 2):
                    problem.addConstraint(equality, [f"h1{y}{st}", f"h{nx+1}{y}{st}"])
                    problem.addConstraint(equality, [f"h1{y}{st}", f"h1{y+1}{st}"])
                    problem.addConstraint(equality, [f"h1{y}{st}", f"h{nx+1}{y+1}{st}"])
            else:
                # Group all corner columns
                hive = f"h11{st}"
                for x in range(1, nx + 2):
                    for y in range(1, ny + 2):
                        bee = f"h{x}{y}{st}"
                        if bee != hive and (x == 1 or y == 1 or x == nx + 1 or y == ny + 1):
                            problem.addConstraint(equality, [hive, bee])

            # Group 4 constraints: Central columns
            hive = f"h22{st}"
            for x in range(2, nx + 1):
                for y in range(2, ny + 1):
                    bee = f"h{x}{y}{st}"
                    if hive != bee:
                        problem.addConstraint(equality, [hive, bee])

            # Constrain beams
            if not self.reduce_combos:
                # Internal not necessarily equal to external
                # Corner beams along x
                hive = [f"bx11{st}", f"hx11{st}"]
                for x in range(1, nx + 1):
                    for y in [1, ny + 1]:
                        bee = [f"bx{x}{y}{st}", f"hx{x}{y}{st}"]
                        if hive[0] != bee[0]:
                            problem.addConstraint(equality, [hive[0], bee[0]])
                            problem.addConstraint(equality, [hive[1], bee[1]])
                # Corner beams along y
                hive = [f"by11{st}", f"hy11{st}"]
                for y in range(1, ny + 1):
                    for x in [1, nx + 1]:
                        bee = [f"by{x}{y}{st}", f"hy{x}{y}{st}"]
                        if hive[0] != bee[0]:
                            problem.addConstraint(equality, [hive[0], bee[0]])
                            problem.addConstraint(equality, [hive[1], bee[1]])
                # Central beams along x
                if ny > 1:
                    hive = [f"bx12{st}", f"hx12{st}"]
                    for x in range(1, nx + 1):
                        for y in range(2, ny + 1):
                            bee = [f"bx{x}{y}{st}", f"hx{x}{y}{st}"]
                            if hive[0] != bee[0]:
                                problem.addConstraint(equality, [hive[0], bee[0]])
                                problem.addConstraint(equality, [hive[1], bee[1]])
                # Central beams along y
                if nx > 1:
                    hive = [f"by21{st}", f"hy21{st}"]
                    for y in range(1, ny + 1):
                        for x in range(2, nx + 1):
                            bee = [f"by{x}{y}{st}", f"hy{x}{y}{st}"]
                            if hive[0] != bee[0]:
                                problem.addConstraint(equality, [hive[0], bee[0]])
                                problem.addConstraint(equality, [hive[1], bee[1]])
            else:
                # Beams along x, internal equal to external
                hive = [f"bx11{st}", f"hx11{st}"]
                for x in range(1, nx + 1):
                    for y in range(1, ny + 2):
                        bee = [f"bx{x}{y}{st}", f"hx{x}{y}{st}"]
                        if bee[0] != hive[0]:
                            problem.addConstraint(equality, [hive[0], bee[0]])
                            problem.addConstraint(equality, [hive[1], bee[1]])
                # Beams along y, internal equal to external
                hive = [f"by11{st}", f"hy11{st}"]
                for y in range(1, ny + 1):
                    for x in range(1, nx + 2):
                        bee = [f"by{x}{y}{st}", f"hy{x}{y}{st}"]
                        if bee[0] != hive[0]:
                            problem.addConstraint(equality, [hive[0], bee[0]])
                            problem.addConstraint(equality, [hive[1], bee[1]])

        # Constrain equality of beam and column sections by creating groups of 2 per storey
        for st in range(1, self.nst, 2):
            # If nst is odd, the last storey will be in a group of 1, so no equality constraint is applied
            for x in range(1, nx + 2):
                for y in range(1, ny + 2):
                    problem.addConstraint(equality, [f"h{x}{y}{st}", f"h{x}{y}{st+1}"])
                    if x < nx + 1:
                        problem.addConstraint(equality, [f"bx{x}{y}{st}", f"bx{x}{y}{st+1}"])
                        problem.addConstraint(equality, [f"hx{x}{y}{st}", f"hx{x}{y}{st+1}"])
                    if y < ny + 1:
                        problem.addConstraint(equality, [f"by{x}{y}{st}", f"by{x}{y}{st+1}"])
                        problem.addConstraint(equality, [f"hy{x}{y}{st}", f"hy{x}{y}{st+1}"])
            # Force allowable variations of c-s dimensions between elements of adjacent groups
            if st <= self.nst - 2:
                for x in range(1, nx + 2):
                    for y in range(1, ny + 2):
                        problem.addConstraint(storey_constraint, [f"h{x}{y}{st}", f"h{x}{y}{st+2}"])
                        if x < nx + 1:
                            problem.addConstraint(storey_constraint, [f"bx{x}{y}{st}", f"bx{x}{y}{st+2}"])
                            problem.addConstraint(storey_constraint, [f"hx{x}{y}{st}", f"hx{x}{y}{st+2}"])
                        if y < ny + 1:
                            problem.addConstraint(storey_constraint, [f"by{x}{y}{st}", f"by{x}{y}{st+2}"])
                            problem.addConstraint(storey_constraint, [f"hy{x}{y}{st}", f"hy{x}{y}{st+2}"])

        # Constrain beam width equal to external column heights connecting the beam
        for st in range(1, self.nst + 1):
            # Along x
            for y in range(1, ny + 2):
                problem.addConstraint(equality, [f"bx1{y}{st}", f"h1{y}{st}"])
            # Along y
            for x in range(1, nx + 2):
                problem.addConstraint(equality, [f"by{x}1{st}", f"h{x}1{st}"])

        # Constrain allowable variation of beam cross-section width and height
        for st in range(1, self.nst + 1):
            for x in range(1, nx + 2):
                for y in range(1, ny + 2):
                    if x < nx + 1:
                        problem.addConstraint(beam_constraint, [f"bx{x}{y}{st}", f"hx{x}{y}{st}"])
                    if y < ny + 1:
                        problem.addConstraint(beam_constraint, [f"by{x}{y}{st}", f"hy{x}{y}{st}"])

        # Constrain beam cross-sections to be equal on a straight line (along each axis)
        # Beams along x axis
        for st in range(1, self.nst + 1):
            for y in range(1, ny + 2):
                hive = [f"bx1{y}{st}", f"hx1{y}{st}"]
                for x in range(2, nx + 1):
                    bee = [f"bx{x}{y}{st}", f"hx{x}{y}{st}"]
                    problem.addConstraint(equality, [hive[0], bee[0]])
                    problem.addConstraint(equality, [hive[1], bee[1]])

        # Beams along y axis
        for st in range(1, self.nst + 1):
            for x in range(1, nx + 2):
                hive = [f"by{x}1{st}", f"hy{x}1{st}"]
                for y in range(2, ny + 1):
                    bee = [f"by{x}{y}{st}", f"hy{x}{y}{st}"]
                    problem.addConstraint(equality, [hive[0], bee[0]])
                    problem.addConstraint(equality, [hive[1], bee[1]])

        # Constrain variation of column cross-sections along x and y (neighbors, external vs internal)
        for st in range(1, self.nst + 1):
            # Along x direction
            for y in range(1, ny + 2):
                external = f"h1{y}{st}"
                if nx > 1:
                    internal = f"h2{y}{st}"
                    problem.addConstraint(bay_constraint, [internal, external])
            # Along y direction
            for x in range(1, nx + 2):
                external = f"h{x}1{st}"
                if ny > 1:
                    internal = f"h{x}2{st}"
                    problem.addConstraint(bay_constraint, [internal, external])

        # Fin all possible solutions within the constraints specified
        solutions = problem.getSolutions()

        print(f"[SUCCESS] Number of solutions found: {len(solutions)}")

        elements = np.zeros((len(solutions), len(ele_types)))
        cnt = 0
        for ele in ele_types:
            for index, solution in enumerate(solutions):
                elements[index][cnt] = solution[ele]
            cnt += 1
        elements = np.unique(elements, axis=0)
        elements = pd.DataFrame(elements, columns=ele_types)

        return elements

    def check_target_t(self, period, limits, tol=0.01):
        """
        Checks if the target meets the tolerance limits for both principal directions
        :param period: float                Principal modes of periods
        :param limits: list                 Period limits
        :param tol: float                   Tolerance for accuracy
        :return: bool                       Verifies if the initial period is within the period range for both
                                            directions
        """
        if limits[0] - tol <= period <= limits[1] + tol:
            return True
        else:
            return False

    def get_weight(self, props):
        """
        gets structural weight of a solution
        :param props: DataFrame                                 Cross-section dimensions of the structural elements
        :return: float                                          Weight of the structural system
        """
        spans_x = self.ipbsd.spans_x
        spans_y = self.ipbsd.spans_y
        heights = self.ipbsd.heights
        w = 0
        for st in range(1, self.nst + 1):
            for x in range(1, self.nbays_x + 2):
                for y in range(1, self.nbays_y + 2):
                    # Add weight of columns (square columns only)
                    w += self.SWEIGHT * props[f"h{x}{y}{st}"] ** 2 * heights[st-1]
                    # Add weight of beams along x
                    if x < self.nbays_x + 1:
                        w += self.SWEIGHT * props[f"bx{x}{y}{st}"] * props[f"hx{x}{y}{st}"] * spans_x[x-1]
                    # Add weight of beams along y
                    if y < self.nbays_y + 1:
                        w += self.SWEIGHT * props[f"by{x}{y}{st}"] * props[f"hy{x}{y}{st}"] * spans_y[y-1]

        return w

    def find_optimal_solution(self, solution=None):
        """
        finds optimal solution based on minimizing weight
        :param solution: Series                                 Solution to run analysis instead (for iterations)
        :return optimal: Series                                 Optimal solution based on minimizing weight
        :return opt_modes: dict                                 Periods and normalized modal shapes of the optimal
                                                                solution
        """
        if solution is None:
            # A new approach to take the case with lowest period, from the loop of cases with lowest weight
            # It tries to ensure that the actual period will be at a more tolerable range
            solutions = self.solutions.nsmallest(20, "Weight")
            optimal = solutions[solutions["T1"] == solutions["T1"].min()].iloc[0]
        else:
            if isinstance(solution, int):
                # ID of solution has been provided, select from existing dataframe by index
                optimal = self.solutions.loc[solution]
            else:
                optimal = solution
                if isinstance(optimal, pd.DataFrame):
                    optimal = optimal.iloc[solution.first_valid_index()]

        cs = self.get_section(optimal)
        # Run modal analysis via OpenSeesPy
        hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}
        op = OpenSeesRun3D(self.ipbsd, cs, self.fstiff, system="space", hinge=hinge)
        op.create_model()
        op.define_masses()
        num_modes = self.nst if self.nst <= 9 else 9
        periods, modalShape, gamma, mstar = op.ma_analysis(num_modes)
        op.wipe()

        weight = self.get_weight(optimal)
        optimal["T1"] = periods[0]
        optimal["T2"] = periods[1]
        optimal["Weight"] = weight
        optimal["Mstar1"] = mstar[0]
        optimal["Mstar2"] = mstar[1]
        optimal["Part Factor1"] = abs(gamma[0])
        optimal["Part Factor2"] = abs(gamma[1])

        opt_modes = {"Periods": periods, "Modes": modalShape}
        return optimal, opt_modes


if __name__ == "__main__":
    import timeit
    from pathlib import Path
    from client.input import Input

    path = Path.cwd()
    period_limits = {"1": [0.313, 0.729], "2": [0.313, 0.614]}
    fstiff = 0.5
    cache_dir = path.parents[1] / ".applications/LOSS Validation Manuscript/space/Cache"/"solution_cache_space.csv"

    input_file = path.parents[1] / ".applications/LOSS Validation Manuscript/space/ipbsd_input.csv"
    data = Input()
    data.read_inputs(input_file)

    start_time = timeit.default_timer()
    cs = CrossSectionSpace(data, period_limits, fstiff, cache_dir=cache_dir)

    elapsed = timeit.default_timer() - start_time
    print('Running time: ', int(elapsed * 10**1) / 10**1, ' seconds')
    print('Running time: ', int(elapsed / float(60) * 10**2) / 10**2, ' minutes')
