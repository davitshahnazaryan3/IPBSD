"""
verifies mean annual frequency of collapse (MAFC) condition
optimization function to automatically select a Say satisfying the MAFC condition
"""
import numpy as np
from scipy.stats import lognorm
from scipy.interpolate import interp1d


class MAFCCheck:
    def __init__(self, r, lam_target, gamma, Hs, sa_haz, omega, hazard):
        """
        initialize mafc check
        :param r: float                             R values of collapse capacity (84th, 50th, 16th percentiles)
        :param lam_target: float                    Target MAFC
        :param gamma: float                         First-mode participation factor
        :param Hs: array                            List of values of annual probability of exceedance
        :param sa_haz: array                        List of corresponding intensities for Hs
        :param omega: float                         Overstrength factor
        :param hazard: str                          Hazard to use, i.e. True or Fitted
        """
        self.say = None
        self.r = r
        self.lam_target = lam_target
        self.gamma = gamma
        self.Hs = Hs
        self.sa_haz = sa_haz
        self.omega = omega
        self.hazard = hazard

    def mafe_direct_im_based(self, eta, beta):
        """
        Details:
        Compute the MAFE of a limit state defined via a fitted lognormal
        distribution by integrating directly with the  hazard curve
        Treat the hazard input to avoid errors.
        We strip out:
         1. the negative H values (usually at the beginning)
         2. the points with constant s (usually at the end)
        Information:
        Author: Gerard J. O'Reilly
        First Version: April 2020
        Notes:
        References:
        Porter KA, Beck JL, Shaikhutdinov R V. Simplified Estimation of Economic
        Seismic Risk for Buildings. Earthquake Spectra 2004; 20(4):
        1239–1263. DOI: 10.1193/1.1809129.
        Inputs:
        :param eta: float                           Fragility function median (intensity)
        :param beta: float                          Fragility function dispersion (total)
        :return: float                              Mean annual frequency of exceedance
        """

        # Do first strip
        s_f = []
        H_f = []
        for aa, bb in zip(self.sa_haz, self.Hs):
            if bb > 0:
                s_f.append(aa)
                H_f.append(bb)

        # Do second strip
        s_ff = []
        H_ff = []
        for i in range(len(s_f) - 1):
            if H_f[i] - self.Hs[i + 1] > 0:
                s_ff.append(s_f[i])
                H_ff.append(H_f[i])
        s_ff.append(s_f[-1])
        H_ff.append(H_f[-1])

        # Overwrite the initial variable for convenience
        s = s_ff
        H = H_ff

        # First we compute the PDF value of the fragility at each of the discrete
        # hazard curve points
        p = lognorm.cdf(s, beta, scale=eta)

        # This function computes the MAF using Method 1 outlined in
        # Porter et al. [2004]
        # This assumes that the hazard curve is linear in logspace between
        # discrete points among others

        # Initialise some arrays
        ds = []
        ms = []
        dHds = []
        dp = []
        dl = []

        for i in np.arange(len(s) - 1):
            ds.append(s[i + 1] - s[i])
            ms.append(s[i] + ds[i] * 0.5)
            dHds.append(np.log(H[i + 1] / H[i]) / ds[i])
            dp.append(p[i + 1] - p[i])
            dl.append(p[i] * H[i] * (1 - np.exp(dHds[i] * ds[i])) - dp[i] / ds[i] * H[i] * (
                        np.exp(dHds[i] * ds[i]) * (ds[i] - 1 / dHds[i]) + 1 / dHds[i]))

        # Compute the MAFE
        l = sum(dl)
        return l

    def objective(self, say):
        """
        objective function to identify yield Sa for optimization
        :param say: float                           Spectral acceleration at yield for an ESDOF
        :return: float                              Difference of target and calculated MAFC
        """
        # transform R to Sa
        self.say = np.array(say)
        # defining scale and standard deviation of lognormal distribution
        mu_lnr = self.say[0]*self.r[1]*self.gamma*self.omega
        std_lnr = np.log(self.r[1]*self.say[0]*self.gamma) - np.log(min(self.r[0], self.r[2])*self.say[0]*self.gamma)

        if self.hazard == "Fitted":
            sa_range = np.linspace(0, 50, 1000)*self.say[0]*self.gamma
            log_dist = lognorm.cdf(sa_range, std_lnr, 0, mu_lnr)
            interpolator = interp1d(sa_range, log_dist)
            new_dist = np.array([])
            sa_new = np.array([])
            for sa in self.sa_haz:
                if max(sa_range) >= sa >= min(sa_range):
                    sa_new = np.append(sa_new, sa)
                    new_dist = np.append(new_dist, interpolator(sa))

            sa_step = self.sa_haz[1] - self.sa_haz[0]
            slopes = abs(np.gradient(self.Hs, sa_step))
            slopes_new = np.array([])
            for i in range(len(self.sa_haz)):
                if self.sa_haz[i] in sa_new:
                    slopes_new = np.append(slopes_new, slopes[i])
            lam = np.trapz(y=np.array(new_dist)*np.array(slopes_new), x=np.array(sa_new))

            return self.lam_target-lam

        elif self.hazard == "True":

            lam = self.mafe_direct_im_based(mu_lnr, std_lnr)
            return self.lam_target - lam
