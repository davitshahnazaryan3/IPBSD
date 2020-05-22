"""
Performs lateral load analysis of frame based on Muto's method
Muto K., Seismic analysis of reinforced concrete buildings
"""
import numpy as np


class ELF:
    def __init__(self, solution, loads, heights, widths):
        """
        Initializes ELF
        :param solution: DataFrame                  Cross-section dimensions of structural elements
        :param loads: list                          Lateral loads
        :param heights: list                        Heights of the frame
        :param widths: list                         Widths of the frame
        """
        self.loads = loads
        self.solution = solution
        self.heights = heights
        self.widths = widths
        self.nbays = len(self.widths)
        self.nst = len(self.heights)
        columns, beams = self.get_properties()
        self.get_D_values(columns, beams)

    def get_properties(self):
        """
        Gets properties necessary for the initiation of Muto's method
        :return: ndarray                            Properties of structural elements including dimensions and stiffness
        """
        columns = np.zeros((self.nst, self.nbays + 1))
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                if bay == 0 or bay == self.nbays:
                    columns[st][bay] = self.solution[f"he{st+1}"]*self.solution[f"he{st+1}"]**3/12/self.heights[st]
                else:
                    columns[st][bay] = self.solution[f"hi{st+1}"]*self.solution[f"hi{st+1}"]**3/12/self.heights[st]
        beams = np.zeros((self.nst, self.nbays))
        for st in range(self.nst):
            for bay in range(self.nbays):
                if bay == 0 or bay == self.nbays - 1:
                    beams[st][bay] = self.solution[f"b{st+1}"]*self.solution[f"h{st+1}"]**3/12/self.widths[0]
                else:
                    beams[st][bay] = self.solution[f"b{st+1}"]*self.solution[f"h{st+1}"]**3/12/self.widths[1]
        return columns, beams

    def get_D_values(self, columns, beams):
        """
        D values are based on the relative stiffness of beams and elements joining at a particular node
        :return:
        """
        k_i = np.zeros((self.nst, self.nbays + 1))
        a_i = np.zeros((self.nst, self.nbays + 1))
        D_values = np.zeros((self.nst, self.nbays + 1))
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                if st == 0:
                    if bay == 0:
                        k_i[st][bay] = beams[st][bay] / columns[st][bay]
                    elif bay == self.nbays + 1:
                        k_i[st][bay] = beams[st][bay-1] / columns[st][bay]
                    else:
                        k_i[st][bay] = (beams[st][bay-1] + beams[st][bay]) / columns[st][bay]
                else:
                    if bay == 0:
                        k_i[st][bay] = (beams[st-1][bay] + beams[st][bay]) / 2 / columns[st][bay]
                    elif bay == self.nbays + 1:
                        k_i[st][bay] = (beams[st-1][bay-1] + beams[st][bay-1]) / 2 / columns[st][bay]
                    else:
                        k_i[st][bay] = (beams[st-1][bay-1] + beams[st-1][bay] + beams[st][bay-1] + beams[st][bay]) / 2 \
                                       / columns[st][bay]

