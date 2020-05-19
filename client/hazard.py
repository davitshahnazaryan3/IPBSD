"""
user defines hazard parameters
"""
import os
import pickle
from external.hazardFit import HazardFit


class Hazard:

    def __init__(self, haz_dir, flname):
        """
        initialize hazard definition or fits the hazard function and generates files for reading
        :param haz_dir: str                                 Hazard directory
        :param flname: str                                  Hazard file name
        :return: None
        """
        self.flname = flname
        self.haz_dir = haz_dir
        self.data_exists = self.check_data()
        if self.data_exists:
            pass
        else:
            HazardFit(self.haz_dir, self.flname, haz_fit=1, pflag=False, save_data=True)

    def check_data(self):
        """
        checks if hazard is already fit
        :return: bool                                       Hazard data exists or not
        """
        data_exists = False
        for file in os.listdir(self.haz_dir):
            if file.startswith("coef"):
                data_exists = True
                break
            else:
                data_exists = False
        return data_exists

    def read_hazard(self):
        """
        reads fitted hazard data
        :return: dataframe, dict                            Coefficients, intensity measures and probabilities of the
                                                            Fitted hazard
                                                            Original hazard data
        """
        with open(self.haz_dir / f"coef_{self.flname}", 'rb') as file:
            coefs = pickle.load(file)
        with open(self.haz_dir / f"fit_{self.flname}", 'rb') as file:
            hazard_data = pickle.load(file)
        with open(self.haz_dir / self.flname, 'rb') as file:
            original_hazard = pickle.load(file)

        return coefs, hazard_data, original_hazard
