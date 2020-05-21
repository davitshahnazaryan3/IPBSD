"""
A software to obtain Moment-curvature relationship of an RC element
Follows the recommendations of Prestressed concrete structures by Collins
Written to optimize for a reinforcement by Davit Shahnazaryan

Units
m for length
kN for forces
mpa for stress
m2 for area
1/m for curvature
+ for tension
- for compression
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from postprocessing.plasticity import Plasticity
import warnings
warnings.filterwarnings('ignore')


class MomentCurvatureRC:
    def __init__(self, b, h, m_target, nlayers=0, p=0, d=.04, fc_prime=25, fy=415, young_mod_s=200e3,
                 check_reinforcement=False, reinf_test=0, plotting=False, soft_method="Collins", k_hard=1.0):
        """
        init Moment curvature tool
        :param b: float                         Element sectional width
        :param h: float                         Element sectional height
        :param m_target: float                  Target flexural capacity
        :param nlayers: int                     Number of flexural reinforcement layers
        :param p: float                         Axial load
        :param d: float                         Flexural reinforcement cover in m
        :param fc_prime: float                  Concrete compressive strength
        :param fy: float                        Reinforcement yield strength
        :param young_mod_s: float               Young modulus of reinforcement
        :param check_reinforcement: bool        Gets moment for reinforcement provided (True) or applied optimization
                                                for Mtarget (False)
        :param reinf_test: int                  Reinforcement for test todo into kwargs
        :param plotting: bool                   Plotting flag
        :param soft_method: str                 Method for the softening slope calculation
        :param k_hard: float                    Hardening slope of reinforcement (i.e. fu/fy)
        """
        self.b = b
        self.h = h
        self.m_target = m_target
        self.nlayers = nlayers
        self.p = p
        self.d = d
        self.fc_prime = fc_prime
        self.fy = fy
        self.young_mod_s = young_mod_s
        self.EPSSH = 0.008
        self.EPSUK = 0.075
        self.k_hard = k_hard
        self.check_reinforcement = check_reinforcement
        self.reinf_test = reinf_test
        self.plotting = plotting
        self.soft_method = soft_method
        self.mi = np.nan
        self.epss = np.nan
        self.fst = np.nan
        self.phii = np.nan

    def checkMy(self, my, data):
        """
        Returns the first yield index
        :param my: int                          Target argument
        :param data: numpy.ndarray              Data for lookup
        :return: numpy.int64 or numpy.nan       Index of target argument
        """
        if np.where(data >= my)[0].size == 0:
            return np.nan
        else:
            return np.where(data >= my)[0][0]

    def plot_mphi(self, phi, m):
        """
        plotting the moment curvature relationship
        :param phi: numpy.ndarray               Curvature
        :param m: numpy.ndarray                 Moment
        :return: None
        """
        f, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(phi, m, 'b', ls='-')
        plt.ylim([0., max(m) + 50])
        plt.xlim([0., max(phi) + 0.05])
        plt.ylabel('Resisting moment [kNm]')
        plt.xlabel('Curvature [1/m]')
        plt.grid(True, which="both", ls="--", lw=0.5)

    def objective(self, c, data):
        """
        Objective function solving for compressed concrete height
        :param c: numpy.ndarray                 Compressed concrete height
        :param data: list                       Reinforcement characteristics
        :return: float                          Difference between internal and external forces
        """
        epsc = data[0]
        epsc_prime = data[1]
        rebar = data[2]
        # reinforcement properties
        ey = self.fy / self.young_mod_s
        fu = self.k_hard * self.fy

        # Block parameters
        b1 = (4 - epsc / epsc_prime) / (6 - 2 * epsc / epsc_prime)
        a1b1 = (epsc / epsc_prime - 1 / 3 * (epsc / epsc_prime) ** 2)

        # assumption, equal compressive and tensile reinforcement
        if self.nlayers == 0:
            z = np.array([self.h - self.d, self.d])
            rebar = np.array([rebar / 2, rebar / 2])
        elif self.nlayers == 1:
            z = np.array([self.h - self.d, self.h / 2, self.d])
            rebar = np.array([rebar * 3 / 8, rebar * 2 / 8, rebar * 3 / 8])
        elif self.nlayers == 2:
            z = np.array([self.h - self.d, ((self.h - 2 * self.d) / (1 + self.nlayers) + self.d) * 2 - self.d,
                          ((self.h - 2 * self.d) / (1 + self.nlayers) + self.d), self.d])
            rebar = np.array([rebar * 4 / 12, rebar * 2 / 12, rebar * 2 / 12, rebar * 4 / 12])
        else:
            raise ValueError(f"[EXCEPTION] wrong input {self.nlayers} of number of reinforcement layers")

        # Strains
        epss = (c - (self.h - z)) / c * epsc
        # Stresses
        stress = np.zeros(len(z))
        for i in range(len(stress)):
            if abs(epss[i]) <= ey:
                stress[i] = self.young_mod_s * epss[i]
            elif ey < abs(epss[i]) <= self.EPSSH:
                stress[i] = np.sign(epss[i]) * self.fy
            elif self.EPSSH < abs(epss[i]) <= self.EPSUK:
                stress[i] = np.sign(epss[i]) * self.fy + (np.sign(epss[i]) * fu - np.sign(epss[i]) * self.fy) * \
                            np.sqrt((epss[i] - np.sign(epss[i]) * self.EPSSH) / (
                                        np.sign(epss[i]) * self.EPSUK - np.sign(epss[i]) * self.EPSSH))
            else:
                # This is an approximation, generally reinforcement will be sufficient enough not to surpass ultimate
                # strain. However, for calculation purpose this will be left here for now
                stress[i] = np.sign(epss[i]) * self.fy + (np.sign(epss[i]) * fu - np.sign(epss[i]) * self.fy) * \
                            np.sqrt((epss[i] - 2 * (np.sign(epss[i]) * self.EPSUK - np.sign(epss[i]) * self.EPSSH)) / (
                                        np.sign(epss[i]) * self.EPSUK - 2 * (np.sign(epss[i]) * self.EPSUK
                                                                             - np.sign(epss[i]) * self.EPSSH)))
            if abs(epss[i]) > self.EPSUK:
                # todo, fix it, not en elegant way of dealing with the problem, in some occasions will be problematic
                stress[i] = 0

        # Forces
        cc = c * a1b1 * self.fc_prime * self.b * 1000
        nslist = rebar * stress * 1000
        nint = cc + sum(nslist)

        self.mi = (cc * (self.h / 2 - b1 * c) + sum(nslist * (z - self.h / 2)))
        self.fst = abs(stress[-1])
        self.epss = abs(epss[-1])
        self.phii = epsc / c
        return abs(nint + self.p)

    def max_moment(self, asi, epsc_prime):
        """
        Gets the maximum moment capacity
        :param asi: numpy.ndarray                   Total reinforcement area
        :param epsc_prime: float                    Concrete strain at peak compressive strength
        :return: float                              Maximum moment of element
        """
        asinit = asi[0]
        c = np.array([0.05])
        c = float(abs(optimize.fsolve(self.objective, c, [2 * epsc_prime, epsc_prime, asinit], factor=0.1)))

        return abs(self.mi / self.k_hard - self.m_target)
    
    def get_softening_slope(self, **kwargs):
        """
        defines the softening slope of the moment-curvature relationship
        :param kwargs: floats                       Total reinforcement area, curvature at yield, axial load ratio
                                                    transverse steel ratio
        :return: float                              Curvature at 0 or residual strength
        """
        if self.soft_method == "Haselton":
            phiy = kwargs.get('curvature_yield', None)
            mu_phi = kwargs.get('curvature ductility', None)
            nu = kwargs.get('axial_load_ratio', 0)
            ro_sh = kwargs.get('transverse_steel_ratio', None)
            if ro_sh is not None:
                theta_pc = 0.76*0.031**nu*(0.02 + 40*ro_sh)**1.02
            else:
                theta_pc = 0.1
            lp = Plasticity().get_lp(db=20, fy=self.fy)
            # todo, fix phi_pc formula, the accuracy needs to be increased as it does not account for elastic portion
            phi_pc = theta_pc/lp
            phi_critical = phiy*mu_phi + phi_pc

        elif self.soft_method == "Collins":
            
            def obj(c, data):
                eps_t = data[0]
                epsc_prime = data[1]
                rebar_area = data[2]
                
                epsc = c*eps_t/(self.h - self.d - c)
                eps_c = (self.d - c)*epsc/c
                a1b1 = (epsc / epsc_prime - 1 / 3 * (epsc / epsc_prime) ** 2)
                C = c*a1b1*self.fc_prime*self.b/1000
                T = -eps_c*self.young_mod_s*rebar_area/1000
                nint = C + T
                return abs(nint + self.p)

            rebar_area = kwargs.get('rebar_area', None)
            young_modulus_rc = (3320 * np.sqrt(self.fc_prime) + 6900)
            n = .8 + self.fc_prime / 17
            epsc_prime = self.fc_prime / young_modulus_rc * n / (n - 1)
            eps_t = self.EPSUK + self.EPSUK*0.1
            c = np.array([0.01])
            c = abs(float(optimize.fsolve(obj, c, [eps_t, epsc_prime, rebar_area], factor=0.1)))
            epsc = c*eps_t/(self.h - self.d - c)
            phi_critical = epsc/c
        else:
            raise ValueError("[EXCEPTION] Wrong method for the definition of softening slope!")
        return phi_critical
    
    def get_mphi(self):
        """
        Gives the Moment-curvature relationship
        :return: dict                              M-phi response data, reinforcement and concrete data for detailing
        """
        # Concrete properties
        # Assumption - parabolic stress-strain relationship for the concrete
        # concrete elasticity modulus MPa
        young_modulus_rc = (3320 * np.sqrt(self.fc_prime) + 6900)
        # young_modulus_rc = 22*((fc_prime+8)/10)**.3*1000
        n = .8 + self.fc_prime / 17
        k_parameter = 0.67 + self.fc_prime/62
        epsc_prime = self.fc_prime / young_modulus_rc * n / (n - 1)
        # Reinforcement properties (500C grade)
        ey = self.fy / self.young_mod_s
        area = self.h * self.b
        inertia = self.b * self.h ** 3 / 12
        # Cracking moment calculation
        lam_nw = 1  # for normal weight concrete
        fcr = 0.33 * lam_nw * np.sqrt(self.fc_prime)
        m_cr = (-self.p / area + fcr * 1000) * inertia / (self.h / 2)
        epscr = fcr / young_modulus_rc
        fcr_t = (-self.p / area + m_cr * self.h / 2 / inertia) / 1000
        yc = fcr * self.h / (fcr + fcr_t)
        phicr = epscr / yc

        ''' The "Process" '''
        epsc = np.linspace(epsc_prime * 2 / 500, 2 * epsc_prime, 1000)
        sigma_c = self.fc_prime*n*epsc/epsc_prime / (n - 1 + np.power(epsc/epsc_prime, n*k_parameter))
        m = np.zeros(len(epsc))
        phi = np.zeros(len(epsc))
        sigmat = np.zeros(len(epsc))
        eps_tensile = np.zeros(len(epsc))

        # Optimize for longitudinal reinforcement at peak capacity
        asinit = np.array([0.001])
        asinit = abs(float(optimize.fsolve(self.max_moment, asinit, epsc_prime, factor=0.1)))
        if self.check_reinforcement:
            c = np.array([0.01])
            c = abs(float(optimize.fsolve(self.objective, c, [2 * epsc_prime, epsc_prime, self.reinf_test], factor=0.1)))
            return self.mi
        else:
            for i in range(len(epsc)):
                # compressed section height optimization
                c = np.array([0.01])
                c = abs(float(optimize.fsolve(self.objective, c, [epsc[i], epsc_prime, asinit], factor=0.1)))

                # tensile reinforcement strains
                eps_tensile[i] = self.epss
                # tensile reinforcement stresses
                sigmat[i] = self.fst
                # bending moment capacity
                m[i] = self.mi
                # curvature
                phi[i] = self.phii

            yield_index = self.checkMy(self.fy, sigmat)

            my_first = m[yield_index]
            phiy_first = phi[yield_index]
            my_nom = self.m_target
            m = np.array(m)
            rpeak = max(m) / my_nom
            ei_cracked = my_first / phiy_first
            phiy_nom = my_nom / ei_cracked
            mu_phi = phi[np.nanargmax(m)] / phiy_nom
            ei_cracked = ei_cracked/young_modulus_rc*self.b*self.h**3/12*1000

        # Softening slope
        phi_critical = self.get_softening_slope(rebar_area=asinit)
        m = np.append(m, 0.0)
        phi = np.append(phi, phi_critical)

        # Plotting
        if self.plotting:
            self.plot_mphi(phi, m)

        # Storing the results
        data = {'curvature': phi, 'moment': m, 'curvature ductility': mu_phi, 'peak/yield ratio': rpeak,
                'reinforcement': asinit, 'cracked EI': ei_cracked, 
                'nominal yield moment': my_nom, 'nominal yield curvature': phiy_nom}
        reinforcement = {"Strain": eps_tensile, "Stress": sigmat}
        concrete = {"Strain": epsc, "Stress": sigma_c}

        return data, reinforcement, concrete

    
if __name__ == '__main__':
    """
    --- Info on the input data:
    b                       section width [m]
    h                       section height [m]
    m_target                target Moment demand [kNm]
    nlayers                 number of reinforcement layers
    p                       external axial force, negative=compressive [kN]
    d                       reinforcement cover [m]
    fc_prime                concrete strength [MPa]
    fy                      reinforcement yield strength [MPa]
    young_modulus_s         reinforcement elastic modulus [MPa]
    check_reinforcement     check for a given reinforcement [bool]
    reinf_test              given reinforcement [m2]
    plotting                plot the M-phi [bool]
    """
    # Section properties
    b = 0.25
    h = 0.25
    Mtarget = 220.

    a = MomentCurvatureRC(b, h, Mtarget, nlayers=0, plotting=True, d=0.02, soft_method="Collins")
    x, r, c = a.get_mphi()
    aa = a.get_mphi()
    ro = x['reinforcement'] / b / (h - 20 / 1000) * 100 / 2
    # print(f"Reinforcement ratio {ro:.2f}%")
    # print(f"Curvature ductility {x['curvature ductility']:.1f}")
#    fig, ax = plt.subplots(figsize=(4, 3))
#    plt.plot(c["Strain"], c["Stress"])
#
#    fig, ax = plt.subplots(figsize=(4, 3))
#    plt.plot(r["Strain"], r["Stress"])
