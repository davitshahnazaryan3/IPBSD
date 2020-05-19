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

    def capacity_design(self):
        """
        applies capacity design strong column - weak beam concept
        :return: lists                      Beam and column yield moments
        """
        Mbi = []
        for st in range(self.nst):
            Mbi.append(max([self.demands["Beams"][i * self.nst + st]["M"] for i in range(self.nbays)]))
        Mbi = self.nbays * Mbi
        Mci = []
        for bay in range(self.nbays+1):
            for st in range(self.nst):
                if bay == 0 or bay == self.nbays:
                    Mci.append(min([self.demands["Columns"][i * self.nst + st]["M"] for i in [0, self.nbays]]))
                else:
                    Mci.append(min([self.demands["Columns"][i * self.nst + st]["M"] for i in range(1, self.nbays)]))
        Myc = np.zeros(self.nst*(self.nbays + 1))
        Myb = Mbi.copy()
        cnt = self.nst*(self.nbays+1)-1
        for j in range(self.nbays, -1, -1):
            for i in range(self.nst-1, -1, -1):
                if i != self.nst-1:
                    if j == 0:
                        if (Mci[i] + Myc[cnt+1])/Myb[i] < 1.3:
                            Myc[cnt] = 1.3*Myb[i]-Myc[cnt+1]
                        else:
                            Myc[cnt] = Mci[i]
                    elif j == self.nbays:
                        if (Mci[j*self.nst+i] + Myc[cnt+1])/Myb[self.nst*self.nbays-self.nst+i] < 1.3:
                            Myc[cnt] = 1.3*Myb[self.nst*self.nbays-self.nst+i]-Myc[cnt+1]
                        else:
                            Myc[cnt] = Mci[j*self.nst+i]
                    else:
                        if (Mci[j*self.nst+i] + Myc[cnt+1])/(Myb[(j-1)*self.nst+i] + Myb[j*self.nst+i]) < 1.3:
                            Myc[cnt] = 1.3*(Myb[(j-1)*self.nst+i] + Myb[j*self.nst+i])-Myc[cnt+1]
                        else:
                            Myc[cnt] = Mci[j*self.nst+i]
                else:
                    if j == 0:
                        if (Mci[i])/Myb[i] < 1.3:
                            Myc[cnt] = max(1.3*Myb[i], Mci[i])
                        else:
                            Myc[cnt] = Mci[i]
                    elif j == self.nbays:
                        if (Mci[j*self.nst+i])/Myb[self.nst*self.nbays-self.nst+i] < 1.3:
                            Myc[cnt] = max(1.3*Myb[self.nst*self.nbays-self.nst+i], Mci[j*self.nst+i])
                        else:
                            Myc[cnt] = Mci[j*self.nst+i]
                    else:
                        if (Mci[j*self.nst+i])/(Myb[(j-1)*self.nst+i] + Myb[j*self.nst+i]) < 1.3:
                            Myc[cnt] = max(1.3*(Myb[(j-1)*self.nst+i] + Myb[j*self.nst+i]), Mci[j*self.nst+i])
                        else:
                            Myc[cnt] = Mci[j*self.nst+i]
                cnt -= 1
        return Myb, Myc

    def design_elements(self):
        """
        designs elements using demands from ELFM and optimal solution, uses moment_curvature_rc
        :return: dict                           Designed element properties from the moment-curvature relationship
        """
        myb, myc = self.capacity_design()
        data = {"Beams": {}, "Columns": {}}
        # Beams
        for st in range(self.nst):
            m_target = myb[st]
            b = self.sections[f"b{st+1}"]
            h = self.sections[f"h{st+1}"]
            mphi = MomentCurvatureRC(b, h, m_target, d=self.rebar_cover)
            data["Beams"][st] = mphi.get_mphi()

        # Columns
        cnt = 0
        for bay in range(0, 2):
            for st in range(self.nst):
                if bay == 0:
                    b = h = self.sections[f"he{st+1}"]
                    m_target = myc[st]
                    nc_design = min([self.demands["Columns"][i * self.nst + st]["N"] for i in [0, self.nbays]])
                else:
                    b = h = self.sections[f"hi{st+1}"]
                    m_target = myc[st+self.nst]
                    nc_design = min([self.demands["Columns"][i * self.nst + st]["N"] for i in range(1, self.nbays)])
                nlayers = 0 if h <= 0.35 else 1 if (0.35 < h <= 0.55) else 2
                mphi = MomentCurvatureRC(b, h, m_target, p=nc_design, nlayers=nlayers, d=self.rebar_cover)
                data["Columns"][cnt] = mphi.get_mphi()
                cnt += 1
        return data
