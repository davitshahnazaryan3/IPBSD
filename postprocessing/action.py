"""
derives base shear and/or lateral forces to act as design action forces
"""
import numpy as np
import pandas as pd


class Action:
    def __init__(self, solution, n_seismic, nbays, nst, masses, say, df, analysis, gravity_loads, num_modes=None,
                 opt_modes=None, modal_sa=None):
        """

        :param solution: DataFrame          Solution containing cross-section and modal properties
        :param n_seismic: int               Number of seismic frames
        :param nbays: int                   Number of bays
        :param masses: array                Lumped storey masses
        :param say: float                   Spectral acceleration at yield
        :param df: DataFrame                SLS table generated through Transformations object
        :param analysis: int                Analysis type
        :param gravity_loads: dict          Gravity loads as {'roof': *, 'floor': *}
        :param num_modes: int               Number of modes to consider for SRSS
        :param opt_modes: dict              Periods and normalized modal shapes of the optimal solution
        :param modal_sa: list               Spectral acceleration to be used for RMSA
        """
        self.solution = solution
        self.n_seismic = n_seismic
        self.nbays = nbays
        self.nst = nst
        self.masses = masses
        self.say = say
        self.df = df
        self.analysis = analysis
        self.gravity_loads = gravity_loads
        self.num_modes = num_modes
        self.opt_modes = opt_modes
        self.modal_sa = modal_sa
        # Gravitational acceleration in m/s2
        self.g = 9.81

    def get_vb(self):
        """
        Gets design base shear
        :return: float                      Design base shear
        """
        mstar = self.solution["Mstar"]
        gamma = self.solution["Part Factor"]
        vb = self.say*mstar*self.n_seismic*gamma*9.81
        vb_i = vb/self.n_seismic
        return vb_i

    def forces(self):
        """
        gets the lateral forces
        :return: DataFrame                  Lateral forces for ELFM
        """
        if self.analysis == 1 or self.analysis == 2 or self.analysis == 3:
            d = pd.DataFrame({'phi': np.array(self.df.loc['phi']),
                              'm': [mi / self.n_seismic for mi in self.masses],
                              'Fi': [0] * self.nst,
                              'Vi': [0] * self.nst})
            # lateral forces
            for n in range(self.nst):
                d.at[n, 'Fi'] = d['m'][n]*self.df.loc['phi'][n] * self.get_vb()/sum(map(lambda x, y: x * y, list(d['m']),
                                                                                        list(self.df.loc['phi'])))
            # base shear at each storey level
            for n in range(self.nst):
                d.at[n, 'Vi'] = sum(fi for fi in d['Fi'][n:self.nst])

            # Check for gravity loads
            if self.analysis == 3:
                grav_loads = np.array([])
                for n in range(self.nst):
                    if n == self.nst - 1:
                        grav_loads = np.append(grav_loads, self.gravity_loads['roof'])
                    else:
                        grav_loads = np.append(grav_loads, self.gravity_loads['floor'])
                d["G"] = grav_loads
            return d

        elif self.analysis == 4 or self.analysis == 5:
            if self.num_modes is None:
                self.num_modes = self.nst
            if self.opt_modes is None:
                self.num_modes = 1
                self.opt_modes = np.array(self.df.loc['phi'])
                self.modal_sa = np.array(self.say)

            masses = self.masses/self.n_seismic
            M = np.zeros([self.nst, self.nst])
            for st in range(self.nst):
                M[st][st] = masses[st]
            identity = np.ones((1, self.nst))

            modes = self.opt_modes["Modes"]
            periods = self.opt_modes["Periods"]
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
                grav_loads = np.array([])
                for n in range(self.nst):
                    if n == self.nst - 1:
                        grav_loads = np.append(grav_loads, self.gravity_loads['roof'])
                    else:
                        grav_loads = np.append(grav_loads, self.gravity_loads['floor'])
                d["G"] = grav_loads
            return d

        else:
            raise ValueError("[EXCEPTION] Wrong type of analysis for definition of action selected")
