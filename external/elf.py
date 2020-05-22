"""
Performs lateral load analysis of frame based on Muto's method
Muto K., Seismic analysis of reinforced concrete buildings
"""
import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d


class ELF:
    def __init__(self, solution, loads, heights, widths):
        """
        Initializes ELF
        :param solution: Series                     Cross-section dimensions of structural elements
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
        self.table = pd.read_csv("external/mutos_table.csv")
        columns, beams = self.get_properties()
        d_values, k_i = self.get_D_values(columns, beams)
        self.response = self.get_internal_forces(d_values, k_i, beams)

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
        :param columns: ndarray                     Stiffness of columns, i.e. I/L
        :param beams: ndarray                       Stiffness of beams, i.e. I/L
        :return D_values: ndarray                   D values
        :return k_i: ndarray                        Relative stiffness factors
        """
        k_i = np.zeros((self.nst, self.nbays + 1))
        a_i = np.zeros((self.nst, self.nbays + 1))
        D_values = np.zeros((self.nst, self.nbays + 1))
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                if st == 0:
                    if bay == 0:
                        k_i[st][bay] = beams[st][bay] / columns[st][bay]
                    elif bay == self.nbays:
                        k_i[st][bay] = beams[st][bay-1] / columns[st][bay]
                    else:
                        k_i[st][bay] = (beams[st][bay-1] + beams[st][bay]) / columns[st][bay]
                    a_i[st][bay] = (0.5 + k_i[st][bay])/(2 + k_i[st][bay])
                else:
                    if bay == 0:
                        k_i[st][bay] = (beams[st-1][bay] + beams[st][bay]) / 2 / columns[st][bay]
                    elif bay == self.nbays:
                        k_i[st][bay] = (beams[st-1][bay-1] + beams[st][bay-1]) / 2 / columns[st][bay]
                    else:
                        k_i[st][bay] = (beams[st-1][bay-1] + beams[st-1][bay] + beams[st][bay-1] + beams[st][bay]) / 2 \
                                       / columns[st][bay]
                    a_i[st][bay] = k_i[st][bay]/(2 + k_i[st][bay])
                D_values[st][bay] = a_i[st][bay]*columns[st][bay]*10**6
        return D_values, k_i

    def get_internal_forces(self, D_values, k_i, beams):
        """
        Gets internal forces in beams and columns
        :param D_values: ndarray                    D values
        :param k_i: ndarray                         Relative stiffness factors
        :param beams: ndarray                       Stiffness of beams
        :return: dict                               Internal forces of beams and columns
        """
        # Storey shear in kN
        v_storey = np.zeros(len(self.loads))
        for i in range(len(self.loads)):
            if i != 0:
                v_storey[i] = self.loads[i]
            else:
                v_storey[i] = self.loads[i-1] + self.loads[i]

        # Column internal forces
        m_columns = np.zeros((self.nst, self.nbays + 1))
        v_columns = np.zeros((self.nst, self.nbays + 1))
        m_col_up = np.zeros((self.nst, self.nbays + 1))
        m_col_low = np.zeros((self.nst, self.nbays + 1))
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                y0 = self.get_y0(st+1, k_i[st][bay])
                # Shear forces in kN
                v_columns[st][bay] = D_values[st][bay]/sum(D_values[st])*v_storey[st]
                # Bending moments at both ends of the column in kNm
                m_col_up[st][bay] = v_columns[st][bay]*self.heights[st]*(1-y0)
                m_col_low[st][bay] = v_columns[st][bay]*self.heights[st]*y0
                # Critical column bending moment in kNm
                m_columns[st][bay] = max(m_col_up[st][bay], m_col_low[st][bay])

        # Beam internal forces
        m_beams = np.zeros((self.nst, self.nbays))
        v_beams = np.zeros((self.nst, self.nbays))
        m_beams_left = np.zeros((self.nst, self.nbays))
        m_beams_right = np.zeros((self.nst, self.nbays))
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                # Left joints
                if bay == 0:
                    if st != self.nst - 1:
                        m_beams_left[st][bay] = m_col_up[st][bay] + m_col_low[st+1][bay]
                    else:
                        m_beams_left[st][bay] = m_col_up[st][bay]
                # Right joints
                elif bay == self.nbays:
                    if st != self.nst - 1:
                        m_beams_right[st][bay-1] = m_col_up[st][bay] + m_col_low[st+1][bay]
                    else:
                        m_beams_right[st][bay-1] = m_col_up[st][bay]
                # Interior joints
                else:
                    if st != self.nst - 1:
                        m_beams_right[st][bay-1] = (m_col_up[st][bay] + m_col_low[st+1][bay]) /\
                                                   (beams[st][bay-1]/beams[st][bay] + 1)
                        m_beams_left[st][bay] = m_col_up[st][bay] + m_col_low[st+1][bay] - m_beams_right[st][bay-1]
                    else:
                        m_beams_right[st][bay-1] = m_col_up[st][bay] / (beams[st][bay-1]/beams[st][bay] + 1)
                        m_beams_left[st][bay] = m_col_up[st][bay] - m_beams_right[st][bay-1]
        for st in range(self.nst):
            for bay in range(self.nbays + 1):
                if bay != self.nbays:
                    # Critical bending moment in kNm
                    m_beams[st][bay] = max(m_beams_right[st][bay], m_beams_left[st][bay])
                    x = m_beams_right[st][bay]*self.widths[bay]/(m_beams_right[st][bay] + m_beams_left[st][bay])
                    # Shear forces in kN
                    v_beams[st][bay] = m_beams_left[st][bay]/x

        # Getting the axial forces
        n_columns = np.zeros((self.nst, self.nbays + 1))
        for bay in range(self.nbays, -1, -1):
            v = 0.
            for st in range(self.nst-1, -1, -1):
                if bay == 0:
                    v += v_beams[st][bay]
                    n_columns[st][bay] = v
                if bay == self.nbays:
                    v += v_beams[st][bay-1]
                    n_columns[st][bay] = v
                else:
                    v += abs(v_beams[st][bay] - v_beams[st][bay-1])
                    n_columns[st][bay] = v
        response = {"Beams": {"M": m_beams, "V": v_beams}, "Columns": {"M": m_columns, "V": v_columns, "N": n_columns}}
        return response

    def get_y0(self, n, ki):
        """
        Gets distance from the location of the point of inflection of the element
        Based on a table generated assuming triangular distribution of lateral loads.
        :param n: int                       Storey of column under consideration
        :param ki: float                    Sum of beam stiffness over sum of column stiffness at the joint
        :return: float                      Factor of distance from the location of the point of inflection
        """
        def round_up(num, decimals=0):
            multiplier = 10**decimals
            return math.ceil(num*multiplier) / multiplier

        def round_down(num, decimals=0):
            multiplier = 10**decimals
            return math.floor(num*multiplier) / multiplier

        if ki >= 1:
            k_low = int(round_down(ki, decimals=0))
            k_up = int(round_up(ki, decimals=0))
        else:
            k_low = round_down(ki, decimals=1)
            k_up = round_up(ki, decimals=1)
            if k_up >= 1.0:
                k_up = int(round(k_up, 0))
        k_list = [k_low, k_up]
        data = self.table[(self.table["m"] == self.nst) & (self.table["n"] == n)]
        y_list = []
        for i in range(len(k_list)):
            y_list.append(float(data[str(k_list[i])]))
        interpolation = interp1d(k_list, y_list)
        y0 = interpolation(ki)
        return y0
