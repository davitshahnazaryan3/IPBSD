"""
derives base shear and/or lateral forces to act as design action forces
"""
import numpy as np
import pandas as pd


class Action:
    def __init__(self, data, analysis, gravity_loads, num_modes=None, opt_modes=None, modal_sa=None):
        """
        :param data: dict                   Input parameters to IPBSD
        :param analysis: int                Analysis type
        :param gravity_loads: dict          Gravity loads as {'roof': *, 'floor': *}
        :param num_modes: int               Number of modes to consider for SRSS
        :param opt_modes: dict              Periods and normalized modal shapes of the optimal solution
        :param modal_sa: list               Spectral acceleration to be used for RMSA
        """
        self.n_seismic = data.n_seismic                 # Number of seismic frames
        self.nst = data.nst                             # Number of storeys
        self.masses = data.masses                       # Lumped storey masses
        self.analysis = analysis
        self.gravity_loads = gravity_loads
        self.num_modes = num_modes
        self.opt_modes = opt_modes
        self.modal_sa = modal_sa
        self.pdelta_loads = data.pdelta_loads           # Gravity loads over P-Delta columns as {'roof': *, 'floor': *}
        # Gravitational acceleration in m/s2
        self.g = 9.81

    def get_vb(self, cy, mstar, part_factor):
        """
        Gets design base shear (for Space systems Vb is for the entire system)
        :param cy: float                    Spectral acceleration at yield
        :param mstar: float                 1st Mode participation mass
        :param part_factor: float           1st Mode participation factor
        :return: float                      Design base shear
        """
        vb = cy * mstar * self.n_seismic * part_factor * self.g
        vb_i = vb / self.n_seismic
        return vb_i

    def forces(self, solution, df, cy):
        """
        gets the lateral forces
        :param solution: DataFrame          Solution containing cross-section and modal properties
        :param df: DataFrame                SLS table generated through Transformations object
        :param cy: float                    Spectral acceleration at yield
        :return: DataFrame                  Lateral forces for ELFM
        """
        if self.analysis == 1 or self.analysis == 2 or self.analysis == 3:
            # TODO, use phi shape from actual modal analysis of the nonlinear system if available
            d = pd.DataFrame({'phi':    np.array(df.loc['phi']),
                              'm':      [mi / self.n_seismic for mi in self.masses],
                              'Fi':     [0] * self.nst,
                              'Vi':     [0] * self.nst,
                              'pdelta': [pdelta / self.n_seismic for pdelta in self.pdelta_loads]})

            # lateral forces
            for n in range(self.nst):
                d.at[n, 'Fi'] = d['m'][n] * df.loc['phi'][n] * self.get_vb(cy, solution['Mstar'],
                                                                           solution['Part Factor']) / \
                                sum(map(lambda x, y: x * y, list(d['m']), list(df.loc['phi'])))

            # base shear at each storey level
            for n in range(self.nst):
                d.at[n, 'Vi'] = sum(fi for fi in d['Fi'][n:self.nst])

            # Check for gravity loads
            if self.analysis == 3:
                d["G"] = self.get_gravity_loads()
            return d

        elif self.analysis == 4 or self.analysis == 5:
            if self.num_modes is None:
                self.num_modes = self.nst
            if self.opt_modes is None:
                self.num_modes = 1
                self.opt_modes = np.array(df.loc['phi'])
                self.modal_sa = np.array(cy)

            masses = self.masses/self.n_seismic
            M = np.zeros([self.nst, self.nst])
            for st in range(self.nst):
                M[st][st] = masses[st]
            identity = np.ones((1, self.nst))

            # Modal parameters
            modes = self.opt_modes["Modes"]
            part_factor = np.zeros(self.num_modes)
            modes_transposed = modes.transpose()
            for mode in range(self.num_modes):
                modes = modes_transposed[0: self.nst, mode: mode+1]
                part_factor[mode] = (modes.transpose().dot(M)).dot(identity.transpose()) / \
                                    (modes.transpose().dot(M)).dot(modes)

            # Generating the action
            forces = np.zeros([self.nst, self.num_modes])
            # lateral loads
            lat_forces = np.zeros([self.nst, self.num_modes])
            for mode in range(self.num_modes):
                modes = modes_transposed[0: self.nst, mode: mode+1]
                forces[0: self.nst, mode: mode+1] = part_factor[mode] * (M.dot(modes))
                lat_forces[0: self.nst, mode: mode+1] = forces[0: self.nst, mode: mode+1]*self.modal_sa[mode]*self.g
            d = {"Fi": lat_forces}

            # Check for gravity loads
            if self.analysis == 5:
                d["G"] = self.get_gravity_loads()
            return d

        else:
            raise ValueError("[EXCEPTION] Wrong type of analysis for definition of action selected")

    def get_gravity_loads(self):
        if self.gravity_loads is not None:
            grav_loads = np.array([])
            for n in range(self.nst):
                if n == self.nst - 1:
                    grav_loads = np.append(grav_loads, self.gravity_loads['roof'])
                else:
                    grav_loads = np.append(grav_loads, self.gravity_loads['floor'])
            return grav_loads
        else:
            return None
