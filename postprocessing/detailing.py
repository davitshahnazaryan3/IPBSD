"""
defines detailing conditions (code-based) for element design
The detailing phase comes as a phase before an iteration where the SPO curve needs to be updated
"""
from external.momentcurvaturerc import MomentCurvatureRC
from postprocessing.plasticity import Plasticity
import numpy as np


class Detailing:
    def __init__(self, demands, nst, nbays, fy, fc, bay_widths, heights, n_seismic, mi, tlower, tupper, dy, sections,
                 rebar_cover=0.03, ductility_class="DCM", young_mod_s=200e3, k_hard=1.0):
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
        :param dy: float                    System yield displacement in m
        :param sections: DataFrame          Cross-sections of elements of the solution
        :param rebar_cover: float           Reinforcement cover in m
        :param ductility_class: str         Ductility class (DCM or DCH, following Eurocode 8 recommendations)
        :param young_mod_s: float           Young modulus of reinforcement
        :param k_hard: float                Hardening slope of reinforcement (i.e. fu/fy)
        """
        self.demands = demands
        self.nst = nst
        self.nbays = nbays
        self.fy = fy
        self.young_mod_s = young_mod_s
        self.fc = fc
        self.bay_widths = bay_widths
        self.heights = heights
        self.n_seismic = n_seismic
        self.mi = mi
        self.tlower = tlower
        self.tupper = tupper
        self.dy = dy
        self.sections = sections
        self.rebar_cover = rebar_cover
        self.ductility_class = ductility_class
        # Reinforcement characteristic yield strength in MPa
        self.FYK = 500.
        self.k_hard = k_hard

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

                elif option == "mean":
                    Mbi[st][0] = Mbi[st][self.nbays-1] = np.mean((Mbi[st][0], Mbi[st][self.nbays-1]))
                    Mbi[st][1:self.nbays-1] = np.mean(Mbi[st][1:self.nbays-1])
                    Mci[st][0] = Mci[st][self.nbays] = np.mean((Mci[st][0], Mci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Mci[st][bay] = Mci[st][-bay-1] = np.mean((Mci[st][bay], Mci[st][-bay-1]))
                    Nci[st][0] = Nci[st][self.nbays] = np.mean((Nci[st][0], Nci[st][self.nbays]))
                    for bay in range(1, int((self.nbays-1)/2)):
                        Nci[st][bay] = Nci[st][-bay-1] = np.mean((Nci[st][bay], Nci[st][-bay-1]))

                elif option == "min":
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

    def ensure_local_ductility(self, b, h, reinforcement, relation, st, bay, eletype):
        """
        Local ductility checks according to Eurocode 8
        :param b: float                             Width of element
        :param h: float                             Height of element
        :param reinforcement: float                 Total reinforcement area
        :param relation: class                      Object created based on M-phi relationship
        :param st: int                              Storey level
        :param bay: int                             Bay level
        :param eletype: str                         Element type, beam or column
        :return: dict                               M-phi outputs to be stored
        """
        # Behaviour factor, for frame systems assuming regularity in elevation
        # Assuming multi-storey, multi-bay frames
        if self.ductility_class == "DCM":
            q = 3.9
        elif self.ductility_class == "DCH":
            q = 5.85
        else:
            raise ValueError("[EXCEPTION] Wrong type of ductility class, must be DCM or DCH!")
        # Design compressive strength
        fcd = 0.8/1.5*self.fc
        # Tensile strength
        fctm = 0.3*self.fc**(2/3)

        # Reinforcement ratio
        ro_prime = reinforcement / (b * (h - self.rebar_cover))
        req_duct = q * 2 - 1
        yield_strain = self.fy / self.young_mod_s

        # Reinforcement ratio limits to meet local ductility conditions
        if eletype == "Beam":
            ro_max = ro_prime + 0.0018 * fcd / self.fy / req_duct / yield_strain
            ro_min = 0.5 * fctm / self.FYK
        elif eletype == "Column":
            ro_min = 0.01
            ro_max = 0.04
        else:
            raise ValueError("[EXCEPTION] Wrong type of element for local ductility verifications!")

        # Verifications
        if ro_min > ro_prime:
            ro_prime = ro_min
            if eletype == "Beam":
                rebar = (b * (h - self.rebar_cover)) * ro_prime*2
            else:
                rebar = (b * (h - self.rebar_cover)) * ro_prime
            m_target = relation.get_mphi(check_reinforcement=True, reinf_test=rebar)
            data = relation.get_mphi(m_target=m_target)
            return data

        elif ro_max < ro_prime:
            print(f"[WARNING] Cross-section of {eletype} element at storey {st} and bay {bay} should be increased! "
                  f"ratio: {ro_prime*100:.2f}%")
            return None

        else:
            return None

    def design_elements(self, modes=None):
        """
        designs elements using demands from ELFM and optimal solution, uses moment_curvature_rc
        :param modes: dict                      Periods and modal shapes obtained from modal analysis
        :return: dict                           Designed element details from the moment-curvature relationship
        """
        # Ensure symmetry of strength distribution along the widths of the frame
        mbi, mci, nci = self.ensure_symmetry(option="max")
        myb, myc = self.capacity_design(mbi, mci)
        data = {"Beams": {}, "Columns": {}}
        # Design of beams
        for st in range(self.nst):
            if self.nbays > 2:
                for bay in range(int(round(self.nbays/2, 0))):      # todo, check whether it should be round up or down
                    m_target = myb[st][bay]
                    b = self.sections[f"b{st+1}"]
                    h = self.sections[f"h{st+1}"]
                    mphi = MomentCurvatureRC(b, h, m_target, d=self.rebar_cover, young_mod_s=self.young_mod_s,
                                             k_hard=self.k_hard)
                    data["Beams"][f"S{st+1}B{bay+1}"] = mphi.get_mphi()
                    '''Local ductility requirement checks (following Eurocode 8 recommendations)'''
                    d_temp = self.ensure_local_ductility(b, h, data["Beams"][f"S{st+1}B{bay+1}"][0]["reinforcement"]/2,
                                                         mphi, st+1, bay+1, eletype="Beam")
                    if d_temp is not None:
                        data["Beams"][f"S{st+1}B{bay+1}"] = d_temp

            else:
                m_target = myb[st][0]
                b = self.sections[f"b{st + 1}"]
                h = self.sections[f"h{st + 1}"]
                mphi = MomentCurvatureRC(b, h, m_target, d=self.rebar_cover, young_mod_s=self.young_mod_s,
                                         k_hard=self.k_hard)
                data["Beams"][f"S{st+1}B{1}"] = mphi.get_mphi()
                '''Local ductility requirement checks (following Eurocode 8 recommendations)'''
                d_temp = self.ensure_local_ductility(b, h, data["Beams"][f"S{st+1}B{1}"][0]["reinforcement"]/2, mphi,
                                                     st+1, 1, eletype="Beam")
                if d_temp is not None:
                    data["Beams"][f"S{st+1}B{1}"] = d_temp

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
                # Assuming contraflexure at 0.6 of height
                # todo, may add better estimation of contraflexure point based on Muto's approach
                z = 0.6*self.heights[st]
                mphi = MomentCurvatureRC(b, h, m_target, length=z, p=nc_design, nlayers=nlayers, d=self.rebar_cover,
                                         young_mod_s=self.young_mod_s, k_hard=self.k_hard)
                data["Columns"][f"S{st+1}B{bay+1}"] = mphi.get_mphi()
                '''Local ductility requirement checks (following Eurocode 8 recommendations)'''
                d_temp = self.ensure_local_ductility(b, h, data["Columns"][f"S{st+1}B{bay+1}"][0]["reinforcement"], mphi,
                                                     st+1, bay+1, eletype="Column")
                if d_temp is not None:
                    data["Columns"][f"S{st+1}B{bay+1}"] = d_temp

        mu_c = self.get_hardening_ductility(data, modes)

        return data, mu_c

    def get_hardening_ductility(self, details, modes):
        """
        Gets hardening ductility
        :param details: dict                    Moment-curvature relationships of the elements
        :param modes: dict                      Periods and modal shapes obtained from modal analysis
        :return: float                          Hardening ductility
        """
        p = Plasticity(lp_name="Priestley", db=20, fy=self.fy, fu=self.fy*self.k_hard, lc=self.heights)
        mu_c = p.get_hardening_ductility(self.dy, details, modes)
        return mu_c

    def get_fracturing_ductility(self, mu_c, sa_c, sa_f, theta_pc, theta_y):
        """
        Gets fracturing ductility
        :param mu_c: float                      System hardening ductility
        :param sa_c: float                      System peak spectral acceleration capacity
        :param sa_f: float                      System residual spectral acceleration capacity
        :param theta_pc: float                  Column post-capping rotation capacity
        :param theta_y: float                   Column yield rotation capacity
        :return: float                          System fracturing ductility
        """
        p = Plasticity()
        mu_f = p.get_fracturing_ductility(mu_c, sa_c, sa_f, theta_pc, theta_y)
        return mu_f
