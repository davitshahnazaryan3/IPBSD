"""
identifies feasible period range
"""
from scipy.interpolate import interp1d
import numpy as np


class PeriodRange:
    def __init__(self, delta_d, alpha_d, sd, sa):
        """
        Initialize
        :param delta_d: float                   Design spectral displacement
        :param alpha_d: float                   Design spectral acceleration
        :param sd: array                        Spectral displacements at SLS
        :param sa: array                        Spectral accelerations at SLS
        """
        self.delta_d = delta_d
        self.alpha_d = alpha_d
        self.sd = sd
        self.sa = sa

    def get_new_spectra(self):
        """
        gets the new spectral values for interpolation
        :return: array, array                   Spectral displacements and accelerations
        """
        if True in np.isnan(self.sa):
            new_sa_sls = np.delete(self.sa, np.arange(np.argwhere(np.isnan(self.sa))[0][0], len(self.sa), 1))
            new_sd_sls = np.delete(self.sd, np.arange(np.argwhere(np.isnan(self.sa))[0][0], len(self.sa), 1))
            max_sa_ind = np.where(new_sa_sls == max(new_sa_sls))[0][0]
            max_sd_ind = np.where(new_sd_sls == max(new_sd_sls))[0][0]
            new_sa_sls = new_sa_sls[max_sa_ind:max_sd_ind + 1]
            new_sd_sls = new_sd_sls[max_sa_ind:max_sd_ind + 1]
        else:
            max_sa_ind = np.where(self.sa == max(self.sa))[0][0]
            max_sd_ind = np.where(self.sd == max(self.sd))[0][0]
            new_sa_sls = self.sa[max_sa_ind:max_sd_ind + 1]
            new_sd_sls = self.sd[max_sa_ind:max_sd_ind + 1]

        return new_sa_sls, new_sd_sls

    def get_T_lower(self, new_sa_sls, new_sd_sls):
        """
        gets the lower bound of T
        :param new_sa_sls: array                Spectral accelerations
        :param new_sd_sls: array                Spectral displacements
        :return: float                          Lower period bound
        """
        interp_sa = interp1d(new_sa_sls, new_sd_sls)
        if self.alpha_d > max(new_sa_sls):
            Sd_1 = float(interp_sa(max(new_sa_sls)))
            T_lower = 2 * np.pi * np.sqrt(Sd_1 / self.alpha_d / 9.81 / 100)

        else:
            Sd_1 = float(interp_sa(self.alpha_d))
            T_lower = 2 * np.pi * np.sqrt(Sd_1 / self.alpha_d / 9.81 / 100)
        return T_lower

    def get_T_upper(self, new_sa_sls, new_sd_sls):
        """
        gets the upper bound of T
        :param new_sa_sls: array                Spectral accelerations
        :param new_sd_sls: array                Spectral displacements
        :return: float                          Upper period bound
        """
        interp_sd = interp1d(new_sd_sls, new_sa_sls)
        if self.delta_d*100 > max(new_sd_sls):
            Sa_1 = float(interp_sd(max(new_sd_sls)))
            T_upper = 2 * np.pi*np.sqrt(self.delta_d*100 / Sa_1 / 9.81 / 100)

        else:
            Sa_1 = float(interp_sd(self.delta_d*100))
            T_upper = 2 * np.pi*np.sqrt(self.delta_d*100 / Sa_1 / 9.81 / 100)
        return T_upper
