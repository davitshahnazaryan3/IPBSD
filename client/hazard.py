"""
user defines hazard parameters
"""
import os
import pickle
from external.hazardFit import HazardFit


class Hazard:

    def __init__(self, flname, outputPath):
        """
        initialize hazard definition or fits the hazard function and generates files for reading
        :param flname: str                                  Hazard file name
        :param outputPath: str                              Outputs path
        :return: None
        """
        self.flname = flname
        self.outputPath = outputPath
        self.data_exists = self.check_data()
        if self.data_exists:
            pass
        else:
            HazardFit(self.flname, self.outputPath, haz_fit=1, pflag=False, save_data=True)

    def check_data(self):
        """
        checks if hazard is already fit
        :return: bool                                       Hazard data exists or not
        """
        data_exists = False
        for file in os.listdir(self.outputPath):
            if file.startswith("coef"):
                data_exists = True
                break
            else:
                data_exists = False
        return data_exists

    def read_hazard(self):
        """
        reads fitted hazard data
        :return: Dataframe, dict                            Coefficients, intensity measures and probabilities of the
                                                            Fitted hazard
                                                            Original hazard data
        """
        filename = os.path.basename(self.flname)
        with open(self.outputPath / f"coef_{filename}", 'rb') as file:
            coefs = pickle.load(file)
        with open(self.outputPath / f"fit_{filename}", 'rb') as file:
            hazard_data = pickle.load(file)
        with open(self.flname, 'rb') as file:
            original_hazard = pickle.load(file)

        return coefs, hazard_data, original_hazard
