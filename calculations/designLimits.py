"""
identifies design limits for the verification fo expected annual loss (EAL)
"""
from client.slf import SLF


class DesignLimits:
    def __init__(self, slf_filename, y):
        """
        Initialize SLF reading
        :param slf_filename: str            Provided SLF filename
        :param y: array                     ELRs associated with each component group
        """
        self.slf_filename = slf_filename
        self.y = y
        self.theta_max = None
        self.a_max = None

        self.slf = SLF(self.slf_filename, self.y)
        self.provided()

    def provided(self):
        """
        provided as input SLF function
        """
        y_target, y, interp_func = self.slf.provided_slf()
        self.theta_max = []
        self.a_max = []
        for i in range(len(y_target)):
            theta_psd = [float(interp_func[0](y[0][i])), float(interp_func[1](y[1][i]))]
            a_pfa = float(interp_func[2](y[2][i]))
            self.theta_max.append(round(min(theta_psd), 4))
            self.a_max.append(round(a_pfa, 2))
