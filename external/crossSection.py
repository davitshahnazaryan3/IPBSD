"""
Optimizes for the fundamental period by seeking cross-sections of all structural elements
"""
from external.getT1 import GetT1
import numpy as np
import constraint
import pandas as pd


class CrossSection:
    def __init__(self, nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, fstiff, tlower, tupper, iteration=False):
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
        self.SWEIGHT = 25.
        if not iteration:
            self.elements = self.constraint_function()
            self.solutions = self.get_all_solutions()

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
            weight = self.get_weight(properties)
            period, phi = self.run_ma(properties)

            M = np.zeros((self.nst, self.nst))
            for st in range(self.nst):
                M[st][st] = self.mi[st]/self.n_seismic
            identity = np.ones((1, self.nst))
            gamma = (phi.transpose().dot(M)).dot(identity.transpose()) / (phi.transpose().dot(M)).dot(phi)
            mstar = (phi.transpose().dot(M)).dot(identity.transpose())

            if self.check_target_t(period):
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
        :param ele: pandas series                               Elements for each solution
        :return: arrays                                         Element cross-section dimensions for a given solution
        """
        hce = []
        hci = []
        b = []
        h = []

        for st in range(self.nst):
            hce.append(ele[f'he{st+1}'])
            hci.append(ele[f'hi{st+1}'])
            b.append(ele[f'b{st+1}'])
            h.append(ele[f'h{st+1}'])
        return hce, hci, b, h

    def create_props(self, hce, hci, b, h):
        """
        Creates A, I of section
        :param hce: array                                       Height of external columns
        :param hci: array                                       Height of internal columns
        :param b: array                                         Beam width
        :param h: array                                         Beam height
        :return: arrays                                         Areas and moments of inertia of all possible elements
        """
        # TODO, different beam sections along the height, once the OpenSees model can accommodate it, no grouping
        #  for now
        a_cols = [hce[i]*hce[i] for i in range(self.nst)]
        i_cols = [hce[i]*hce[i]**3/12 for i in range(self.nst)]
        a_cols_int = [hci[i]*hci[i] for i in range(self.nst)]
        i_cols_int = [hci[i]*hci[i]**3/12 for i in range(self.nst)]
        a_beams = [b[i]*h[i] for i in range(self.nst)]
        i_beams = [b[i]*h[i]**3/12 for i in range(self.nst)]
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

        ele_types = []
        problem = constraint.Problem()
        for i in range(self.nst):
            problem.addVariable(f'b{i+1}', np.arange(0.25, 0.55, 0.05))
            problem.addVariable(f'h{i+1}', np.arange(0.40, 0.75, 0.05))
            problem.addVariable(f'hi{i+1}', np.arange(0.25, 0.75, 0.05))
            problem.addVariable(f'he{i+1}', np.arange(0.25, 0.75, 0.05))
            ele_types.append(f'he{i+1}')
            ele_types.append(f'hi{i+1}')
            ele_types.append(f'b{i+1}')
            ele_types.append(f'h{i+1}')

        for i in range(self.nst-1):
            problem.addConstraint(eq_constraint, [f'b{i+1}', f'b{i+2}'])
            problem.addConstraint(eq_constraint, [f'h{i+1}', f'h{i+2}'])
            problem.addConstraint(storey_constraint, [f'hi{i+1}', f'hi{i+2}'])
            problem.addConstraint(storey_constraint, [f'he{i+1}', f'he{i+2}'])
        problem.addConstraint(eq_constraint, [f'b{self.nst}', f'he{self.nst}'])
        problem.addConstraint(beam_constraint, [f'b{self.nst}', f'h{self.nst}'])
        for i in range(self.nst):
            problem.addConstraint(bay_constraint, [f'hi{i+1}', f'he{i+1}'])
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
        checks if the target meets tolerance limits
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
                                 (self.nbays - 1) + props[4][st] * sum(self.bay_widths))
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
            optimal = self.solutions[self.solutions["Weight"] == self.solutions["Weight"].min()].iloc[0]
        else:
            optimal = solution
        hce, hci, b, h = self.get_section(optimal)
        properties = self.create_props(hce, hci, b, h)
        period, phi = self.run_ma(properties, single_mode=False)

        opt_modes = {"Periods": period, "Modes": phi}

        return optimal, opt_modes


if __name__ == "__main__":
    import timeit
    start_time = timeit.default_timer()

    nst = 4
    nbays = 3
    fy = 415
    fc = 25
    bay_widths = [5, 5, 5]
    heights = [3.5, 3, 3, 3]
    n_seismic = 2
    fstiff = 0.5
    mi = [99.08, 99.08, 99.08, 82.57]
    tlower = 0.5
    tupper = 1.0
    ma = CrossSection(nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, fstiff, tlower, tupper)
    opt_sol = ma.find_optimal_solution()
    print(opt_sol)

    def truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    # --------- Stop the clock and report the time taken in seconds
    elapsed = timeit.default_timer() - start_time
    print('Running time: ', truncate(elapsed, 1), ' seconds')
    print('Running time: ', truncate(elapsed / float(60), 2), ' minutes')
