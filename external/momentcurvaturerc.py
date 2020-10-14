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
    def __init__(self, b, h, m_target, length=0, nlayers=0, p=0, d=.03, fc_prime=25, fy=415, young_mod_s=200e3,
                 plotting=False, soft_method="Haselton", k_hard=1.0, fstiff=0.5, AsTotal=None, distAs=None):
        """
        init Moment curvature tool
        :param b: float                         Element sectional width
        :param h: float                         Element sectional height
        :param m_target: float                  Target flexural capacity
        :param length: float                    Distance from critical section to point of contraflexure
        :param nlayers: int                     Number of flexural reinforcement layers
        :param p: float                         Axial load
        :param d: float                         Flexural reinforcement cover in m
        :param fc_prime: float                  Concrete compressive strength
        :param fy: float                        Reinforcement yield strength
        :param young_mod_s: float               Young modulus of reinforcement
        :param plotting: bool                   Plotting flag
        :param soft_method: str                 Method for the softening slope calculation
        :param k_hard: float                    Hardening slope of reinforcement (i.e. fu/fy)
        :param fstiff: float                    Stiffness reduction factor (50% per Eurocode 8), for the model only
        :param AsTotal: float                   Total reinforcement area (for beams only)
        :param distAs: list                     Relative distributions of reinforcement (for beams only)
        """
        self.b = b
        self.h = h
        self.m_target = m_target
        self.length = length
        self.nlayers = nlayers
        self.p = p
        self.d = d
        self.fc_prime = fc_prime
        self.fy = fy
        self.young_mod_s = young_mod_s
        self.EPSSH = 0.008
        self.EPSUK = 0.075
        self.k_hard = k_hard
        self.plotting = plotting
        self.soft_method = soft_method
        self.fstiff = fstiff
        self.mi = np.nan
        self.epss = np.nan
        self.fst = np.nan
        self.phii = np.nan
        self.AsTotal = AsTotal
        self.distAs = distAs
        if self.distAs is None:
            self.distAs = [0.5, 0.5]
        # Transverse reinforcement spacing in [m]
        self.TRANSVERSE_SPACING = 0.1
        # Transverse reinforcement diameter in [m]
        self.TRANSVERSE_DIAMETER = 8/1000
        # Number of transverse reinforcement legs
        self.TRANSVERSE_LEGS = 4

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
            rebar = np.array([rebar*self.distAs[1], rebar*self.distAs[0]])
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
            # if abs(epss[i]) > self.EPSUK:
            #     # todo, fix it, not en elegant way of dealing with the problem, in some occasions will be problematic
            #     stress[i] = 0

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
        lp = Plasticity(lp_name="Priestley", db=20, fy=self.fy, fu=self.fy*self.k_hard, lc=self.length).get_lp()
        if self.soft_method == "Haselton":
            phiy = kwargs.get('curvature_yield', None)
            mu_phi = kwargs.get('curvature_ductility', None)
            nu = kwargs.get('axial_load_ratio', 0)
            ro_sh = kwargs.get('transverse_steel_ratio', None)
            if ro_sh is not None:
                theta_pc = min(0.76*0.031**nu*(0.02 + 40*ro_sh)**1.02, 0.1)
            else:
                theta_pc = 0.1
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
        return phi_critical, lp
    
    def get_mphi(self, check_reinforcement=False, reinf_test=0, m_target=None, reinforcements=None):
        # TODO, a bit too rigid, make it more flexible, easier to manipulate within IPBSD to achieve optimized designs
        """
        Gives the Moment-curvature relationship
        :param check_reinforcement: bool            Gets moment for reinforcement provided (True) or applied
                                                    optimization for Mtarget (False)
        :param reinf_test: int                      Reinforcement for test
        :param m_target: float                      Target bending moment. This is a value that may be increased
                                                    depending on local ductility requirements
        :param reinforcements: list                 Positive and negative reinforcements (for beams only)
        :return: dict                               M-phi response data, reinforcement and concrete data for detailing
        """
        if reinforcements is not None:
            self.AsTotal = sum(reinforcements)
            self.distAs = reinforcements / self.AsTotal
        if m_target is not None:
            self.m_target = m_target

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
        # Cracking moment calculation (irrelevant for the design, but will store the data for possible checks)
        lam_nw = 1  # for normal weight concrete
        fcr = 0.33 * lam_nw * np.sqrt(self.fc_prime)
        m_cr = (-self.p / area + fcr * 1000) * inertia / (self.h / 2)
        epscr = fcr / young_modulus_rc
        fcr_t = (-self.p / area + m_cr * self.h / 2 / inertia) / 1000
        yc = fcr * self.h / (fcr + fcr_t)
        phi_cr = epscr / yc

        ''' The "Process" '''
        epsc = np.linspace(epsc_prime * 2 / 500, 2 * epsc_prime, 1000)
        sigma_c = self.fc_prime*n*epsc/epsc_prime / (n - 1 + np.power(epsc/epsc_prime, n*k_parameter))
        m = np.zeros(len(epsc))
        phi = np.zeros(len(epsc))
        sigmat = np.zeros(len(epsc))
        eps_tensile = np.zeros(len(epsc))

        # Optimize for longitudinal reinforcement at peak capacity
        if self.AsTotal is not None:
            asinit = np.array([self.AsTotal])
        else:
            asinit = np.array([0.001])
            
        asinit = abs(float(optimize.fsolve(self.max_moment, asinit, epsc_prime, factor=0.1)))
        if check_reinforcement:
            c = np.array([0.01])
            self.mi = None
            init_factor = 2.
            while self.mi is None or np.isnan(self.mi[0]):
                c = abs(float(optimize.fsolve(self.objective, c, [init_factor * epsc_prime, epsc_prime, reinf_test],
                                              factor=0.1)))
                init_factor -= 0.1
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

            # Removing None arguments
            if self.k_hard == 1.:
                m = m[~np.isnan(m)]
                phi = phi[~np.isnan(phi)]
            else:
                idx = min(np.argwhere(np.isnan(m))[0][0], np.argwhere(np.isnan(phi))[0][0])
                m = m[:idx]
                phi = phi[:idx]
            idx_max = -1
            m_max = m[idx_max]

            my_first = m[yield_index]
            phiy_first = phi[yield_index]
            m = np.array(m)
            rpeak = m_max / my_first
            ei_cracked = my_first / phiy_first
            mu_phi = phi[idx_max] / phiy_first
            ei_cracked = ei_cracked / (young_modulus_rc * self.b * self.h ** 3 / 12 * 1000)

        # Softening slope
        nu = abs(self.p)/area/self.fc_prime/1000
        ro_sh = self.TRANSVERSE_LEGS*np.pi*self.TRANSVERSE_DIAMETER**2/4 / self.TRANSVERSE_SPACING / self.b
        A_sh = self.TRANSVERSE_LEGS*np.pi*self.TRANSVERSE_DIAMETER**2/4
        phi_critical, lp = self.get_softening_slope(rebar_area=asinit, curvature_yield=phiy_first,
                                                    curvature_ductility=mu_phi, axial_load_ratio=nu,
                                                    transverse_steel_ratio=ro_sh)
        # Identifying fracturing point
        m = np.append(m, 0.0)
        phi = np.append(phi, phi_critical)
        fracturing_ductility = phi_critical/phiy_first

        # Plotting
        if self.plotting:
            self.plot_mphi(phi, m)

        # Storing the results
        # The values are relative to the point of yield definition (herein to first yield)
        # If columns are designed, full reinforcement is recorded, if beams, then the direction of interest is recorded
        As_factor = 1. if self.AsTotal is None else self.distAs[0]

        data = {'curvature': phi, 'moment': m, 'curvature_ductility': mu_phi, 'peak/yield ratio': rpeak,
                'reinforcement': asinit*As_factor, 'cracked EI': ei_cracked, 'first_yield_moment': my_first,
                'first_yield_curvature': phiy_first, 'phi_critical': phi_critical,
                'fracturing_ductility': fracturing_ductility, "lp": lp, "cover": self.d, "A_sh": A_sh,
                "spacing": self.TRANSVERSE_SPACING}
        reinforcement = {"Strain": eps_tensile, "Stress": sigmat}
        concrete = {"Strain": epsc, "Stress": sigma_c}

        # Hysteretic behaviour of all structural elements for model creation in OpenSees (M-curvature)
        # Assuming 50% of gross cross-section (the actual M-phi is calculated without the necessity of defining fstiff)
        # todo, add accounting for residual strength here
        curv_yield = self.m_target/young_modulus_rc/1000/inertia/self.fstiff
        curv_ult = mu_phi*phiy_first
        model = {"yield": {"curvature": curv_yield, "moment": self.m_target},
                 "ultimate": {"curvature": curv_ult, "moment": m_max},
                 "fracturing": {"curvature": phi_critical, "moment": 0}}

        return data, reinforcement, concrete, model

    
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
    plotting                plot the M-phi [bool]
    """
    # Section properties
    b = 0.25
    h = 0.45
    Mtarget = 98.12
    cover = 0.03

    mphi = MomentCurvatureRC(b, h, Mtarget, d=cover, plotting=False, soft_method="Haselton", AsTotal=0.0005, distAs=[0.5, 0.5])

    #    mphi = MomentCurvatureRC(b, h, Mtarget, nlayers=0, plotting=True, d=cover, soft_method="Haselton")
    data = mphi.get_mphi()
    ro = data[0]['reinforcement'] / b / (h - cover) * 100
    print(f"Reinforcement ratio {ro:.2f}%")
    
    # plt.plot(data[0]["curvature"], data[0]["moment"])
    # plt.xlim([0, 0.2])
    # plt.ylim([0, 250])
    # plt.scatter(data[0]["first_yield_curvature"], data[0]["first_yield_moment"])
    # plt.scatter(data[0]["first_yield_curvature"] * data[0]["curvature_ductility"],
    #             data[0]["first_yield_moment"] * data[0]["peak/yield ratio"])

#    fig, ax = plt.subplots(figsize=(4, 3))
#    plt.plot(c["Strain"], c["Stress"])
#
#    fig, ax = plt.subplots(figsize=(4, 3))
#    plt.plot(r["Strain"], r["Stress"])
