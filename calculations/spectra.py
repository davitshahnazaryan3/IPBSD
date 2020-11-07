"""
defines spectra at desired performance limit states
"""
import numpy as np
from scipy.interpolate import interp1d


class Spectra:
    def __init__(self, lam, df, Ts):
        """
        Initialize
        :param lam: float           MAF of exceeding SLS
        :param df: DataFrame        2nd-order hazard fit coefficients
        :param Ts: array            Periods used when fitting the hazard
        """
        self.lam = lam
        self.df = df
        self.Ts = Ts
        # Period range
        self.T_RANGE = np.arange(0, 4 + .01, .01)
        # Arrays of spectral acceleration and displacement, sa in [g], sd in [%]
        self.sa, self.sd = self.get_spectra()

    def get_spectra(self):
        # todo, in future, the function should be flexible to generate as many limit state spectra as necessary
        """
        Gets serviceability limit state spectra (SLS)
        :return: lists              Spectral accelerations and spectral displacements associated with the limit state
        """
        s = np.zeros(len(self.T_RANGE))
        d = np.zeros(len(self.T_RANGE))

        # Interpolation functions based on provided hazard for Period vs coefficients
        interpolator_k0 = interp1d(np.array(self.Ts), np.array(self.df.loc['k0']))
        interpolator_k1 = interp1d(np.array(self.Ts), np.array(self.df.loc['k1']))
        interpolator_k2 = interp1d(np.array(self.Ts), np.array(self.df.loc['k2']))

        for i in range(len(self.T_RANGE)):
            # Period
            T_val = self.T_RANGE[i]
            # 2nd-order hazard fitting coefficients
            k0 = interpolator_k0(T_val)
            k1 = interpolator_k1(T_val)
            k2 = interpolator_k2(T_val)
            # Compute the acceleration
            temp = float(np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(self.lam / k0))) / 2 / k2))
            # Assign the acceleration in [g]
            s[i] = temp

            if T_val > 0.0:
                # Assign the displacement in [cm]
                d[i] = 100 * temp * 9.81 * (float(T_val) / 2 / np.pi) ** 2
        return s, d
