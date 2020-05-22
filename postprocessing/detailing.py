"""
defines detailing conditions (code-based) for element design
The detailing phase comes as a phase before an iteration where the SPO curve needs to be updated
"""
from external.momentcurvaturerc import MomentCurvatureRC
import numpy as np


class Detailing:
    def __init__(self, demands, nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, tlower, tupper, sections,
                 rebar_cover=0.03):
        # todo, add design based on M+N and M-N (currently only M-N)
        """
        initializes detailing phase
        :param demands: dict                Demands on structural elements
        :param nst: int                     Number of stories
        :param nbays: int                   Number of bays
        :param fy: float                    Reinforcement yield strength
        :param fc: float                    Concrete compressive strength
        :param bay_widths: list             Bay widths
        :param heights: list                Storey heights
        :param n_seismic: int               Number of seismic frames
        :param mi: list                     Lumped storey masses
        :param tlower: float                Lower period bound
        :param tupper: float                Upper period bound
        :param sections: DataFrame          Cross-sections of elements of the solution
        :param rebar_cover: float           Reinforcement cover in m
        """
        self.demands = demands
        self.nst = nst
        self.nbays = nbays
        self.fy = fy
        self.fc = fc
        self.bay_widths = bay_widths
        self.heights = heights
        self.n_seismic = n_seismic
        self.mi = mi
        self.tlower = tlower
        self.tupper = tupper
        self.sections = sections
        self.rebar_cover = rebar_cover

    def capacity_design(self, Mbi, Mci):
        """
        applies capacity design strong column - weak beam concept
        :param Mbi: ndarray                 Moment demands on beams
        :param Mci: ndarray                 Moment demands on columns
        :return: ndarray                    Beam and column moment demands
        """
        Myc = np.zeros(Mci.shape)
        Myb = Mbi.copy()
        for bay in range(self.nbays, -1, -1):
            for st in range(self.nst-1, -1, -1):
                # Roof storey level
                if st == self.nst - 1:
                    if bay == 0:
                        if Mci[st][bay] / Myb[st][bay] < 1.3:
                            Myc[st][bay] = max(1.3*Myb[st][bay], Mci[st][bay])
                        else:
                            Myc[st][bay] = Mci[st][bay]
                    elif bay == self.nbays:
                        if Mci[st][bay] / Myb[st][bay-1] < 1.3:
                            Myc[st][bay] = max(1.3*Myb[st][bay-1], Mci[st][bay])
                        else:
                            Myc[st][bay] = Mci[st][bay]
                    else:
                        if Mci[st][bay] / (Myb[st][bay-1] + Myb[st][bay]) < 1.3:
                            Myc[st][bay] = max(1.3*(Myb[st][bay-1] + Myb[st][bay]), Mci[st][bay])
                        else:
                            Myc[st][bay] = Mci[st][bay]
                else:
                    if bay == 0:
                        if (Mci[st][bay] + Mci[st+1][bay]) / Myb[st][bay] < 1.3:
                            Myc[st][bay] = 1.3*Myb[st][bay] - Mci[st+1][bay]
                        else:
                            Myc[st][bay] = Mci[st][bay]
                    elif bay == self.nbays:
                        if (Mci[st][bay] + Mci[st+1][bay]) / Myb[st][bay-1] < 1.3:
                            Myc[st][bay] = 1.3*Myb[st][bay-1] - Mci[st+1][bay]
                        else:
                            Myc[st][bay] = Mci[st][bay]
                    else:
                        if (Mci[st][bay] + Mci[st+1][bay]) / (Myb[st][bay-1] + Myb[st][bay]) < 1.3:
                            Myc[st][bay] = 1.3*(Myb[st][bay-1] + Myb[st][bay]) - Mci[st+1][bay]
                        else:
                            Myc[st][bay] = Mci[st][bay]
        return Myb, Myc

    def ensure_symmetry(self, option="max"):
        """
        Ensures symmetric strength along the width of the frame
        :param option: str                  Technique of reading demands ('max', 'mean', 'min')
        :return: ndarray                    Internal force demands of structural elements, i.e. M of beams,
                                            M and N of columns
        """
        Mbi = self.demands["Beams"]["M"]
        Mci = self.demands["Columns"]["M"]
        Nci = self.demands["Columns"]["N"]
        if self.nbays <= 2:
            for st in range(self.nst):
                if option == "max":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.max((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mci[st][0] = Mci[st][self.nbays] = np.max((Mci[st][0], Mci[st][self.nbays]))
                    Nci[st][0] = Nci[st][self.nbays] = np.max((Nci[st][0], Nci[st][self.nbays]))
                elif option == "mean":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.mean((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mci[st][0] = Mci[st][self.nbays] = np.mean((Mci[st][0], Mci[st][self.nbays]))
                    Nci[st][0] = Nci[st][self.nbays] = np.mean((Nci[st][0], Nci[st][self.nbays]))
                elif option == "min":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.min((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mci[st][0] = Mci[st][self.nbays] = np.min((Mci[st][0], Mci[st][self.nbays]))
                    Nci[st][0] = Nci[st][self.nbays] = np.min((Nci[st][0], Nci[st][self.nbays]))
                else:
                    raise ValueError("[EXCEPTION] Wrong option for ensuring symmetry, must be max, mean or min")
        else:
            for st in range(self.nst):
                if option == "max":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.max((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mbi[st][1:self.nbays-1] = np.max(Mbi[st][1:self.nbays-1])
                    Mci[st][0] = Mci[st][self.nbays] = np.max((Mci[st][0], Mci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Mci[st][bay] = Mci[st][-bay-1] = np.max((Mci[st][bay], Mci[st][-bay-1]))
                    Nci[st][0] = Nci[st][self.nbays] = np.max((Nci[st][0], Nci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Nci[st][bay] = Nci[st][-bay-1] = np.max((Nci[st][bay], Nci[st][-bay-1]))

                if option == "mean":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.mean((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mbi[st][1:self.nbays-1] = np.mean(Mbi[st][1:self.nbays-1])
                    Mci[st][0] = Mci[st][self.nbays] = np.mean((Mci[st][0], Mci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Mci[st][bay] = Mci[st][-bay-1] = np.mean((Mci[st][bay], Mci[st][-bay-1]))
                    Nci[st][0] = Nci[st][self.nbays] = np.mean((Nci[st][0], Nci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Nci[st][bay] = Nci[st][-bay-1] = np.mean((Nci[st][bay], Nci[st][-bay-1]))

                if option == "min":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.min((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mbi[st][1:self.nbays-1] = np.min(Mbi[st][1:self.nbays-1])
                    Mci[st][0] = Mci[st][self.nbays] = np.min((Mci[st][0], Mci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Mci[st][bay] = Mci[st][-bay-1] = np.min((Mci[st][bay], Mci[st][-bay-1]))
                    Nci[st][0] = Nci[st][self.nbays] = np.min((Nci[st][0], Nci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Nci[st][bay] = Nci[st][-bay-1] = np.min((Nci[st][bay], Nci[st][-bay-1]))
                else:
                    raise ValueError("[EXCEPTION] Wrong option for ensuring symmetry, must be max, mean or min")
        return Mbi, Mci, Nci

    def design_elements(self):
        """
        designs elements using demands from ELFM and optimal solution, uses moment_curvature_rc
        :return: dict                           Designed element properties from the moment-curvature relationship
        """
        mbi, mci, nci = self.ensure_symmetry()
        myb, myc = self.capacity_design(mbi, mci)
        data = {"Beams": {}, "Columns": {}}
        # Design of beams
        for st in range(self.nst):
            if self.nbays >= 2:
                for bay in range(int(round(self.nbays/2, 0))):
                    m_target = myb[st][bay]
                    b = self.sections[f"b{st+1}"]
                    h = self.sections[f"h{st+1}"]
                    mphi = MomentCurvatureRC(b, h, m_target, d=self.rebar_cover)
                    data["Beams"][f"S{st+1}B{bay+1}"] = mphi.get_mphi()
            else:
                m_target = myb[st][0]
                b = self.sections[f"b{st + 1}"]
                h = self.sections[f"h{st + 1}"]
                mphi = MomentCurvatureRC(b, h, m_target, d=self.rebar_cover)
                data["Beams"][f"S{st+1}B{1}"] = mphi.get_mphi()

        # Design of columns
        for st in range(self.nst):
            for bay in range(int(round((self.nbays+1)/2, 0))):
                if bay == 0:
                    b = h = self.sections[f"he{st+1}"]
                else:
                    b = h = self.sections[f"hi{st+1}"]
                m_target = myc[st][bay]
                nc_design = nci[st][bay]
                nlayers = 0 if h <= 0.35 else 1 if (0.35 < h <= 0.55) else 2
                z = self.heights[st]
                mphi = MomentCurvatureRC(b, h, m_target, length=z, p=nc_design, nlayers=nlayers, d=self.rebar_cover)
                data["Columns"][f"S{st+1}B{bay+1}"] = mphi.get_mphi()

        return data
