"""
verifies fundamental period (T1) condition
"""
from external.crossSection import CrossSection


class PeriodCheck:
    # todo, check if this check is necessary, currently it is underutilized
    def __init__(self, t, t_lower, t_upper, tol=1e-3):
        """
        initialize period check
        :param t:                           First-mode period
        :param t_lower:                     Lower period bound
        :param t_upper:                     Upper period bound
        :param tol: float                   Tolerance for accuracy
        """
        self.t_lower = t_lower
        self.t_upper = t_upper
        self.t = t
        self.tol = tol
        self.check_t()

    def check_t(self):
        """
        optimizes for T
        :return: None
        """
        if self.t_lower-self.tol <= self.t <= self.t_upper+self.tol:
            print(f"T1 of {self.t} is in a range of {round(self.t_lower,2)} and {round(self.t_upper,2)}")
        else:
            print(f"T1 of {self.t} is NOT in a range of {round(self.t_lower,2)} and {round(self.t_upper,2)}")
