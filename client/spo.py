"""
user defines static pushover parameters
"""


class SPO:
    def __init__(self, data):
        self.mc = data['mc']
        self.a = data['a']
        self.ac = data['ac']
        self.r = data['r']
        self.mf = data['mf']
        self.pw = data['pw']
        self.period = data['T']
