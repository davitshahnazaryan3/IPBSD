"""
Defines post-yield properties of structural elements
"""
import numpy as np


class Plasticity:
    def __init__(self, lp_name=None, **kwargs):
        """
        Initializes estimation of plastic properties
        :param lp_name: str                     Plastic hinge length name
        :param kwargs:                          Arguments necessary for the lp calculation method
        """
        self.lp_name = lp_name
        self.kwargs = kwargs

    def get_hardening_ductility(self, dy, details):
        """
        estimates system hardening ductility, based on the knowledge of deltaY, details of columns
        :param dy: float                        System yield displacement
        :param details: dict                    Moment-curvature relationships of the elements
        :return: float                          System hardening ductility
        """
        for i in details["Columns"].keys():
            nst = int(i[1])
        phi_p_list = np.zeros(nst) + 1000.

        for i in details["Columns"].keys():
            phi_y = details["Columns"][i][3]["yield"]["curvature"]
            mu_phi = details["Columns"][i][3]["ultimate"]["curvature"] / phi_y
            phi_u = phi_y * mu_phi
            phi_p = phi_u - phi_y
            if phi_p < phi_p_list[int(i[1])-1]:
                phi_p_list[int(i[1])-1] = phi_p
        phi_p = min(phi_p_list)
        lc_list = self.kwargs.get('lc', None)
        lc = lc_list[np.argmin(phi_p_list)]
        self.kwargs["lc"] = lc
        lp = self.get_lp()
        dp = phi_p*lp*lc
        du = dp + dy
        ductility = du/dy
        return ductility

    def get_theta_pc(self, **kwargs):
        """
        gets column post-capping rotation capacity based on Haselton et al., 2016, DOI: 10.14359/51689245
        :param kwargs: floats                   nu - axial load ratio, ro_sh - transverse reinforcement ratio
        :return: float                          Column post-capping rotation capacity
        """
        nu = kwargs.get("nu", None)
        ro_sh = kwargs.get("ro_sh", None)
        if nu is not None:
            theta_pc = min(0.76*0.031**nu*(0.02 + 40*ro_sh)**1.02, 0.10)
        else:
            theta_pc = 0.10
        return theta_pc

    def get_fracturing_ductility(self, mu_c, sa_c, sa_f, theta_pc, theta_y):
        """
        gets fracturing ductility
        :param mu_c: float                      System hardening ductility
        :param sa_c: float                      System peak spectral acceleration capacity
        :param sa_f: float                      System residual spectral acceleration capacity
        :param theta_pc: float                  Column post-capping rotation capacity
        :param theta_y: float                   Column yield rotation capacity
        :return: float                          System fracturing ductility
        """
        app = (sa_c - sa_f)/(theta_pc/theta_y)
        ductility = mu_c - (sa_c - sa_f)/app
        return ductility

    def get_lp(self):
        """
        gets plastic hinge length
        :param lp_name: str                     Plastic hinge length name
        :param kwargs:                          Arguments necessary for the method
        :return: float                          Plastic hinge length
        """
        if self.lp_name == "Baker":                      # Baker, 1956
            "beams and columns"
            k = self.kwargs.get('k', None)
            z = self.kwargs.get('z', None)
            d = self.kwargs.get('d', None)
            lp = k*(z/d)**(1/4)*d
        elif self.lp_name == "Sawyer":                   # Sawyer, 1964
            z = self.kwargs.get('z', None)
            d = self.kwargs.get('d', None)
            lp = 0.25*d + 0.075*z
        elif self.lp_name == "Corley":                   # Corley, 1966
            "beams"
            z = self.kwargs.get('z', None)
            d = self.kwargs.get('d', None)
            lp = 0.5*d + 0.2*np.sqrt(d)*z/d
        elif self.lp_name == "Mattock":                  # Mattock, 1967
            "beams"
            z = self.kwargs.get('z', None)
            d = self.kwargs.get('d', None)
            lp = 0.5*d + 0.05*z
        elif self.lp_name == "Priestley and Park":       # Priestley and Park, 1987
            "columns"
            z = self.kwargs.get('z', None)
            db = self.kwargs.get('db', None)
            lp = 0.08*z + 6*db
        elif self.lp_name == "Sheikh and Khoury":        # DOI: 10.14359/3960
            "columns under high axial loads"
            h = self.kwargs.get('h', None)
            lp = 1.*h
        elif self.lp_name == "Coleman and Spacone":      # DOI: https://doi.org/10.1061/(ASCE)0733-9445(2001)127:11(1257)
            Gcf = self.kwargs.get('Gcf', None)
            fc_prime = self.kwargs.get('fc_prime', None)
            eps20 = self.kwargs.get('eps20', None)
            epsc = self.kwargs.get('epsc', None)
            young_modulus = self.kwargs.get('young_modulus', None)
            lp = Gcf/(0.6*fc_prime*(eps20 - epsc + 0.8*fc_prime/young_modulus))
        elif self.lp_name == "Panagiotakos and Fardis":  # DOI: 10.14359/10181
            """beams and columns"""
            z = self.kwargs.get('z', None)
            db = self.kwargs.get('db', None)
            fy = self.kwargs.get('fy', None)
            lp = 0.18*z + 0.021*db*fy
        elif self.lp_name == "Bae and Bayrak":           # Bae and Bayrak, 2008
            """columns"""
            h = self.kwargs.get('h', None)
            p = self.kwargs.get('p', None)
            p0 = self.kwargs.get('o0', None)
            As = self.kwargs.get('As', None)
            Ag = self.kwargs.get('Ag', None)
            z = self.kwargs.get('z', None)
            lp = max(h*((0.3*p/p0 + 3*As/Ag - 1)*z/h + 0.25), 0.25*h)
        elif self.lp_name == "Priestley":                # DDBD by Priestley et al., 2007
            """columns"""
            db = self.kwargs.get('db', None)
            lc = self.kwargs.get('lc', None)
            fy = self.kwargs.get('fy', None)
            fu = self.kwargs.get('fu', None)
            lsp = 0.022*fy*db
            if fu != fy:
                k = min(0.2*(fu/fy - 1), 0.08)
            else:
                k = 0.08
            lp = max(2*lsp, k*lc*1000 + lsp)
        else:
            db = self.kwargs.get('db', None)
            fy = self.kwargs.get('fy', None)
            lsp = 0.022 * fy * db
            lp = 2 * lsp
        return lp/1000

    def verify_match(self, x, target, tol=0.05):
        """
        verify if target is met
        :param x: float
        :param target: float
        :param tol: float
        :return: bool
        """
        if x - tol <= target <= x + tol:
            return True
        else:
            return False

    def find_new_solution(self):
        """
        if local ductility not met, modify ro and/or c-s dimensions, get new T1 and estimate hardening ductility
        look in sections for a new section matching the c-s dimensions identified herein
        :return: dataframe
        """
        pass

    def redo_mafc_check(self):
        """
        rerun spo2ida with new hardening ductility, period
        :return: None
        """
        pass
