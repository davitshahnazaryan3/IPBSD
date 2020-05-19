"""
defines hazard function
"""
import numpy as np
import matplotlib.pyplot as plt


class Hazard:
    def __init__(self, coef, cflag, return_period=None, beta_al=None, pflag=False):
        """
        initialize hazard calculation
        :param coef: array                              Hazard 2nd-order fit coefficients
        :param cflag: str                               Intensity measure (SA or PGA)
        :param return_period: array                     Return periods of all limit states considered
        :param beta_al: array                           Uncertainties associated with each limit state considered
        :param pflag: bool                              Plotting flag
        """
        self.Sa_Range = np.linspace(0.005, 5.005, 301)
        self.coef = coef
        self.cflag = cflag
        self.return_period = return_period
        self.beta_al = beta_al
        self.pflag = pflag
        if cflag == "PGA":
            self.lambdaLS, self.PGA = self.get_PGA()
        elif cflag == 'SA':
            self.Hs = self.get_distribution()
            if self.pflag:
                self.plotting()
        else:
            raise ValueError("[EXCEPTION] Wrong call for a hazard function!")

    def get_PGA(self):
        """
        get PGA and MAFE of LS
        :return: array, array                           PGA and MAFE associated with each limit state considered
        """
        k0 = self.coef[0]
        k1 = self.coef[1]
        k2 = self.coef[2]
        H = 1/np.array(self.return_period)
        p = 1/(1+2*k2*(np.power(self.beta_al, 2)))
        lambdaLS = np.sqrt(p) * k0 ** (1 - p) * H ** p * np.exp(0.5 * p * np.power(k1, 2) * (np.power(self.beta_al, 2)))
        PGA = np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(lambdaLS / k0))) / 2 / k2)

        return lambdaLS, PGA

    def get_distribution(self):
        """
        get hazard distribution
        :return: array                                  Hs associated with a limit state of interest
        """
        k0 = self.coef[0]
        k1 = self.coef[1]
        k2 = self.coef[2]
        p = 1/(1+2*k2*(self.beta_al[2]**2))
        Hs = float(k0)*np.exp(-float(k2) * np.log(self.Sa_Range) ** 2 - float(k1) * np.log(self.Sa_Range))
        MAF = np.sqrt(p) * k0 ** (1 - p) * Hs ** p * np.exp(0.5 * p * k1 ** 2 * (self.beta_al[2] ** 2))
        return Hs

    def plotting(self):
        """
        plots the hazard function
        :return: None
        """
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(self.Sa_Range, self.Hs, 'b', ls='--', label='2nd order fit')
        # plt.plot(x_real,Hs_real, 'ro', label = 'SHARE model')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(10e-7, 1)
        plt.xlim(0.1, 1.1)
        plt.xticks([])
        plt.xticks(np.arange(0.1, 1.1, 0.9))
        plt.ylabel(r'$H$(s)')
        plt.xlabel(r'$PGA$ [g]')
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend(frameon=False,
                   loc='upper right',
                   fontsize=12)
