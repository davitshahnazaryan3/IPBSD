"""
defines spectra at desired performance limit states
"""
import numpy as np
from scipy.interpolate import interp1d


class Spectra:
    """
    Initialize
    df and periods are used together, hazard is used separately
    """
    # Period range for the spectrum computation
    T_RANGE = np.arange(0, 4 + .01, .01)

    def get_spectra(self, lam, use_coefs=False, df=None, periods=None, hazard=None):
        """
        Gets serviceability limit state spectra (SLS)
        :param lam: float           MAF of exceeding SLS
        :param use_coefs: bool      True for Fitted hazard, False for True hazard
        :param df: DataFrame        2nd-order hazard fit coefficients
        :param periods: array       Periods used when fitting the hazard
        :param hazard: dict         True hazard data
        :return: lists              Spectral accelerations and spectral displacements associated with the limit state
                                    spectral acceleration and displacement, sa in [g], sd in [%]
        """

        if use_coefs:

            if df is None or periods is None:
                raise ValueError("Fitting coefficients (df) and periods need to be provided!")

            s = np.zeros(len(self.T_RANGE))
            d = np.zeros(len(self.T_RANGE))

            # Unstable, avoid if actual hazard is available
            # Interpolation functions based on provided hazard for Period vs coefficients
            interpolator_k0 = interp1d(np.array(periods), np.array(df.loc['k0']))
            interpolator_k1 = interp1d(np.array(periods), np.array(df.loc['k1']))
            interpolator_k2 = interp1d(np.array(periods), np.array(df.loc['k2']))

            for i in range(len(self.T_RANGE)):
                # Period
                T_val = self.T_RANGE[i]
                # 2nd-order hazard fitting coefficients
                k0 = interpolator_k0(T_val)
                k1 = interpolator_k1(T_val)
                k2 = interpolator_k2(T_val)
                # Compute the acceleration, [g]
                s[i] = float(np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(lam / k0))) / 2 / k2))

                if T_val > 0.0:
                    # Assign the displacement in [cm]
                    d[i] = 100 * s[i] * 9.81 * (float(T_val) / 2 / np.pi) ** 2

        else:
            if hazard is None:
                raise ValueError("True hazard must be provided!")

            # recommended
            self.T_RANGE = np.zeros(len(hazard[0]))
            s = np.zeros(len(hazard[0]))
            d = np.zeros(len(hazard[0]))

            for i in range(len(hazard[0])):
                interpolator = interp1d(np.array(hazard[2][i]), np.array(hazard[1][i]))

                sval = interpolator(lam)
                dval = 100 * sval * 9.81 * (float(i/10) / 2 / np.pi) ** 2

                self.T_RANGE[i] = float(i/10)
                s[i] = sval
                d[i] = dval

        return s, d
