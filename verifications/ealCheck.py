"""
verifies expected annual loss (EAL) condition
"""


class EALCheck:
    def __init__(self, eal, eal_limit):
        """
        Initializes EAL verification
        :param eal: float                                   Actual EAL computed as the area below the loss curve
        :param eal_limit: float                             EAL limit as a performance objective
        """
        self.eal = eal
        self.eal_limit = eal_limit

    def verify_eal(self):
        """
        Verifies if EAL calculated meets the limiting eal
        :return: bool                                       Whether the limit condition is met or not
        """
        if self.eal <= self.eal_limit:
            return True
        else:
            return False
