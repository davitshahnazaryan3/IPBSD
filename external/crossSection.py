"""
Optimizes for the fundamental period by seeking cross-sections of all structural elements
"""
from external.getT1 import GetT1
import numpy as np
import constraint
import pandas as pd


class CrossSection:
    def __init__(self, nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, fstiff, tlower, tupper, iteration=False,
                 cache_dir=None, solution_perp=None):
        """
        Initializes the optimization function for the cross-section for a target fundamental period
        :param nst: int                                     Number of stories
        :param nbays: int                                   Number of bays
        :param fy: float                                    Yield strength of flexural reinforcement
        :param fc: float                                    Concrete compressive strength
        :param bay_widths: array                            Bay widths
        :param heights: array                               Storey heights
        :param n_seismic: int                               Number of seismic frames
        :param mi: array                                    Lumped storey masses
        :param fstiff: float                                Stiffness reduction factor
        :param tlower: float                                Lower period bound
        :param tupper: float                                Upper period bound
        :param iteration: bool                              Whether an iterative analysis is being performed
        :param cache_dir: str                               Directory to export the solution cache if provided
        :param solution_perp: Series                        Solution in perpendicular direction
        """
        self.nst = nst
        self.nbays = nbays
        self.fy = fy
        self.fc = fc
        self.bay_widths = bay_widths
        self.heights = np.array(heights)
        self.n_seismic = n_seismic
        self.fstiff = fstiff
        self.mi = np.array(mi)
        self.tlower = tlower
        self.tupper = tupper
        self.solution_perp = solution_perp
        self.SWEIGHT = 25.

        # Export solution as cache in .csv (Initialize)
        if cache_dir is not None:
            cache_path = cache_dir
            if not iteration and not cache_path.exists():
                self.elements = self.constraint_function()
                self.solutions = self.get_all_solutions()
                # Export solutions as cache in .csv
                self.solutions.to_csv(cache_path)
            # If solutions file exists, read and derive the solutions
            if cache_path.exists():
                self.solutions = pd.read_csv(cache_path, index_col=[0])

    def get_all_solutions(self):
        """
        gets all possible solutions respecting the period bounds
        :return: dict                                       All possible solutions within a period range
        """
        solutions = pd.DataFrame(columns=self.elements.columns)
        solutions["T"] = ""
        solutions["Weight"] = ""
        solutions["Mstar"] = ""
        solutions["Part Factor"] = ""
        cnt = 0
        for i in self.elements.index:
            ele = self.elements.iloc[i]
            hce, hci, b, h = self.get_section(ele)
            properties = self.create_props(hce, hci, b, h)
            period, phi = self.run_ma(properties)

            if self.check_target_t(period):

                weight = self.get_weight(properties)
                M = np.zeros((self.nst, self.nst))
                for st in range(self.nst):
                    M[st][st] = self.mi[st] / self.n_seismic
                identity = np.ones((1, self.nst))
                gamma = (phi.transpose().dot(M)).dot(identity.transpose()) / (phi.transpose().dot(M)).dot(phi)
                mstar = (phi.transpose().dot(M)).dot(identity.transpose())

                solutions = solutions.append(ele, ignore_index=True)
                solutions["T"].iloc[cnt] = period
                solutions["Weight"].iloc[cnt] = weight
                solutions["Part Factor"].iloc[cnt] = gamma
                solutions["Mstar"].iloc[cnt] = mstar
                cnt += 1

        return solutions

    def get_section(self, ele):
        """
        gets all sections
        :param ele: pandas series                               Structural elements of the solution
        :return: arrays                                         Element cross-section dimensions for a given solution
        """
        hce = []
        hci = []
        b = []
        h = []

        for st in range(self.nst):
            hce.append(float(ele[f'he{st+1}']))
            hci.append(float(ele[f'hi{st+1}']))
            b.append(float(ele[f'b{st+1}']))
            h.append(float(ele[f'h{st+1}']))
        hce = np.array(hce)
        hci = np.array(hci)
        b = np.array(b)
        h = np.array(h)
        return hce, hci, b, h

    def create_props(self, hce, hci, b, h):
        """
        Creates cross-section area, A, and moment of inertia, I, of section
        :param hce: array                                       Height of external columns
        :param hci: array                                       Height of internal columns
        :param b: array                                         Beam width
        :param h: array                                         Beam height
        :return: arrays                                         Areas and moments of inertia of all possible elements
        """
        a_cols = hce * hce
        i_cols = hce * hce**3/12
        a_cols_int = hci * hci
        i_cols_int = hci * hci**3/12
        a_beams = np.tile(b * h, (self.nbays, 1))
        i_beams = np.tile(b * h**3/12, (self.nbays, 1))
        return a_cols, a_cols_int, i_cols, i_cols_int, a_beams, i_beams

    def run_ma(self, s_props, single_mode=True):
        """
        runs MA
        :param s_props: tuple of arrays                         Properties of solution elements
        :param single_mode: bool                                Whether to run only for 1st mode or multiple modes
        :return: float, array                                   1st mode period and normalized modal shape
        """
        ma = GetT1(s_props[0], s_props[1], s_props[2], s_props[3], s_props[4], s_props[5], self.nst, self.bay_widths,
                   self.heights, self.mi, self.n_seismic, self.fc, self.fstiff, just_period=True,
                   single_mode=single_mode)
        period, phi = ma.run_ma()
        return period, phi

    def constraint_function(self):
        """
        constraint function for identifying combinations of all possible cross-sections
        :return: DataFrame                                      All solutions with element cross-sections
        """
        # Helper constraint functions
        def storey_constraint(x, y):
            x = round(x, 2)
            y = round(y, 2)
            if x + 10**-5 >= y >= x - 0.05 - 10**-5:
                return True

        def bay_constraint(x, y):
            x = round(x, 2)
            y = round(y, 2)
            if y + 0.2 + 10**-5 >= x >= y - 10**-5:
                return True

        def eq_constraint(x, y):
            x = round(x, 2)
            y = round(y, 2)
            if x + 10**-5 >= y >= x - 10**-5:
                return True

        def beam_constraint(x, y):
            x = round(x, 2)
            y = round(y, 2)
            if x + 0.1 - 10**-5 <= y <= x + 0.3 + 10**-5:
                return True

        # Initialize element types
        ele_types = []
        # Initialize the problem
        problem = constraint.Problem()
        for i in range(self.nst):
            # Limits on cross-section dimensions and types of elements
            if self.solution_perp is not None:
                # Fix external column (as perpendicular or primary direction frame is already found)
                problem.addVariable(f'he{i+1}', np.array([self.solution_perp[f"he{i+1}"]]))
            else:
                # A case where only one direction is being considered
                problem.addVariable(f'he{i+1}', np.arange(0.25, 1.0, 0.05))
            problem.addVariable(f'b{i+1}', np.arange(0.25, 1.0, 0.05))
            problem.addVariable(f'h{i+1}', np.arange(0.40, 1.0, 0.05))
            problem.addVariable(f'hi{i+1}', np.arange(0.25, 1.0, 0.05))
            ele_types.append(f'he{i+1}')
            ele_types.append(f'hi{i+1}')
            ele_types.append(f'b{i+1}')
            ele_types.append(f'h{i+1}')

        for i in range(1, self.nst, 2):
            # Force equality of beam and column sections by creating groups of 2
            # If nst is odd, the last storey will be in a group of 1, so no equality constraint is applied
            problem.addConstraint(eq_constraint, [f'hi{i}', f'hi{i+1}'])
            problem.addConstraint(eq_constraint, [f'he{i}', f'he{i+1}'])
            problem.addConstraint(eq_constraint, [f'b{i}', f'b{i+1}'])
            problem.addConstraint(eq_constraint, [f'h{i}', f'h{i+1}'])
            # Force allowable variations of c-s dimensions between elements of adjacent groups
            if i <= self.nst - 2:
                problem.addConstraint(storey_constraint, [f'hi{i}', f'hi{i+2}'])
                problem.addConstraint(storey_constraint, [f'he{i}', f'he{i+2}'])
                problem.addConstraint(storey_constraint, [f'b{i}', f'b{i+2}'])
                problem.addConstraint(storey_constraint, [f'h{i}', f'h{i+2}'])

        # Force beam width equal to column height
        problem.addConstraint(eq_constraint, [f'b{self.nst}', f'he{self.nst}'])
        # Force allowable variation of beam c-s width and height
        if self.solution_perp is None:
            problem.addConstraint(beam_constraint, [f'b{self.nst}', f'h{self.nst}'])

        if self.solution_perp is None:
            for i in range(self.nst):
                # Force allowable variation of internal and external column c-s dimensions
                # but only for the primary direction
                problem.addConstraint(bay_constraint, [f'hi{i+1}', f'he{i+1}'])

        # Find all possible solutions within the constraints specified
        solutions = problem.getSolutions()

        elements = np.zeros((len(solutions), len(ele_types)))
        cnt = 0
        for ele in ele_types:
            for index, solution in enumerate(solutions):
                elements[index][cnt] = solution[ele]
            cnt += 1
        elements = np.unique(elements, axis=0)
        elements = pd.DataFrame(elements, columns=ele_types)
        return elements

    def check_target_t(self, period, tol=0.01):
        """
        checks if the target meets tolerance limits towards period bounds
        :param period: float                                    1st mode period
        :param tol: float                                       Tolerance for accuracy
        :return: bool                                           Verifies if 1st mode period is within the period range
        """
        if self.tlower - tol <= period <= self.tupper + tol:
            return True
        else:
            return False

    def get_weight(self, props):
        """
        gets structural weight of a solution
        :param props: list                                      Cross-section dimensions of the structural elements
        :return: float                                          Weight of the structural system
        """
        w = 0
        for st in range(self.nst):
            w += self.SWEIGHT * (props[0][st] ** self.heights[st] * 2 + props[1][st] * self.heights[st] *
                                 (self.nbays - 1) + props[4][0][st] * sum(self.bay_widths))
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
            # optimal = self.solutions[self.solutions["Weight"] == self.solutions["Weight"].min()].iloc[0]
            # A new approach to take the case with lowest period, from the loop of cases with lowest weight
            # It tries to ensure that the actual period will be at a more tolerable range
            solutions = self.solutions.nsmallest(20, "Weight")
            optimal = solutions[solutions["T"] == solutions["T"].min()].iloc[0]
        else:
            if isinstance(solution, int):
                # ID of solution has been provided, select from existing dataframe by index
                optimal = self.solutions.loc[solution]
            else:
                optimal = solution
                if isinstance(optimal, pd.DataFrame):
                    optimal = optimal.iloc[solution.first_valid_index()]

        # Cross-section properties of the selected solution
        hce, hci, b, h = self.get_section(optimal)
        properties = self.create_props(hce, hci, b, h)
        period, phi = self.run_ma(properties, single_mode=False)

        # Get modal parameters
        weight = self.get_weight(properties)
        M = np.zeros((self.nst, self.nst))
        for st in range(self.nst):
            M[st][st] = self.mi[st] / self.n_seismic
        identity = np.ones((1, self.nst))
        phi1 = phi[0, :]
        gamma = (phi1.transpose().dot(M)).dot(identity.transpose()) / (phi1.transpose().dot(M)).dot(phi1)
        mstar = (phi1.transpose().dot(M)).dot(identity.transpose())

        optimal["T"] = period[0]
        optimal["Weight"] = weight
        optimal["Part Factor"] = abs(gamma[0])
        optimal["Mstar"] = abs(mstar[0])

        opt_modes = {"Periods": period, "Modes": phi}

        return optimal, opt_modes


if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()
    from pathlib import Path
    mainDir = Path.cwd().parents[0] / "Database"
    
    nst = 3
    nbays = 3
    fy = 415
    fc = 25
    bay_widths = [5, 5, 5]
    heights = [3.5, 3, 3]
    n_seismic = 2
    fstiff = 0.5
    mi = [99.08, 99.08, 82.57]
    tlower = 0.6
    tupper = 0.8
    ma = CrossSection(nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, fstiff, tlower, tupper, cache_dir=mainDir)
    opt_sol = ma.find_optimal_solution()
    print(ma.solutions)

    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    # --------- Stop the clock and report the time taken in seconds
    elapsed = timeit.default_timer() - start_time
    print('Running time: ', truncate(elapsed, 1), ' seconds')
    print('Running time: ', truncate(elapsed / float(60), 2), ' minutes')
