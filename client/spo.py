"""
user defines static pushover parameters
"""


class SPO:
    def __init__(self, data):
        # todo, this should be user input data, decide on data type
        self.mc = data['mc']
        self.a = data['a']
        self.ac = data['ac']
        self.r = data['r']
        self.mf = data['mf']
        self.pw = data['pw']
        self.period = data['T']

    def select_t1(self):
        """

        :return: float or array
        """
        # todo, make it user defined
        #   currently optimal solution comes with the T1, so this function is unnecessary, but this should be an option
        self.period = 1.0

        return self.period
