"""
Defines hazard function
"""
import pickle
import numpy as np
import os

from tools.hazardFit import HazardFit


class Hazard:
    def __init__(self, filename, export_directory, beta_al=None):
        """
        initialize hazard calculation
        :param filename: str                            Hazard file name
        :param export_directory: str                    Export directory
        :param beta_al: array                           Aleatory uncertainties associated with each limit state
        """
        self.Sa_Range = np.linspace(0.005, 5.005, 301)
        self.filename = filename
        self.export_directory = export_directory
        if beta_al:
            self.beta_al = beta_al
        else:
            self.beta_al = [0.1, 0.2, 0.3]

        # if hazard fitting function does not exist, run fitting
        if not self.check_data():
            # this will generate pickle files for reading
            HazardFit(self.filename, self.export_directory, haz_fit=1, export=True)

    def check_data(self):
        """
        checks if hazard is already fit
        :return: bool                                       Hazard data exists or not
        """
        for file in os.listdir(self.export_directory):
            if file.startswith("coef"):
                return True
        return False

    def get_mafe(self, coef, return_period, cflag):
        """
        Get MAFE of LS
        :param coef: array                              Hazard 2nd-order fit coefficients
        :param return_period: array                     Return periods of all limit states considered
        :param cflag: str                               Intensity measure (SA or PGA)
        :return: array, array                           PGA and MAFE associated with each limit state considered
        or
        :return: array                                  Hs associated with a limit state of interest
        """
        k0 = coef[0]
        k1 = coef[1]
        k2 = coef[2]
        p = 1 / (1 + 2*k2*(np.power(self.beta_al, 2)))
        if cflag == "PGA":
            H = 1 / np.array(return_period)
            lambdaLS = np.sqrt(p) * k0 ** (1 - p) * H ** p * np.exp(0.5 * p * np.power(k1, 2) *
                                                                    (np.power(self.beta_al, 2)))
            PGA = np.exp((-k1 + np.sqrt(k1 ** 2 - 4 * k2 * np.log(lambdaLS / k0))) / 2 / k2)
            return lambdaLS
        elif cflag == "SA":
            Hs = float(k0) * np.exp(-float(k2) * np.log(self.Sa_Range) ** 2 - float(k1) * np.log(self.Sa_Range))
            MAF = np.sqrt(p) * k0 ** (1 - p) * Hs ** p * np.exp(0.5 * p * k1 ** 2 * (self.beta_al[2] ** 2))
            return Hs
        else:
            raise ValueError("[EXCEPTION] Wrong call for a hazard function! cflag must be 'PGA' or 'SA'!")

    def read_hazard(self):
        """
        reads fitted hazard data
        :return: Dataframe, dict                            Coefficients, intensity measures and probabilities of the
                                                            Fitted hazard
                                                            True hazard data
        """
        filename = os.path.basename(self.filename)
        with open(self.export_directory / f"coef_{filename}", 'rb') as file:
            coefs = pickle.load(file)
        with open(self.export_directory / f"fit_{filename}", 'rb') as file:
            hazard_data = pickle.load(file)
        with open(self.filename, 'rb') as file:
            true_hazard = pickle.load(file)

        return coefs, hazard_data, true_hazard
