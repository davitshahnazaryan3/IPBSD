"""
user defines storey-loss function parameters
"""
from scipy.interpolate import interp1d
import pandas as pd


class SLF:
    def __init__(self, data, y):
        """
        initialize storey loss function definition
        :param data: dict                   SLF data file
        :param y: array                     Expected loss ratios associated with each component group
        """
        self.data = data
        self.y = y

    def provided_slf(self):
        """
        provided as input SLF function
        :return: array, array, array               Storey loss functions in terms of ELR and EDP
        """
        filename = self.data
        df = pd.read_excel(io=filename)
        pfa_ns_range = df['PFA']
        y_ns_pfa_range = df['E_NS_PFA']
        psd_s_range = df['IDR_S']
        y_s_psd_range = df['E_S_IDR']
        psd_ns_range = df['IDR_NS']
        y_ns_psd_range = df['E_NS_IDR']
        y_psd_range = y_s_psd_range + y_ns_psd_range

        # Normalization of storey loss functions
        max_slf = max(y_s_psd_range) + max(y_ns_psd_range) + max(y_ns_pfa_range)
        max_s_psd = max(y_s_psd_range) / max_slf
        max_ns_psd = max(y_ns_psd_range) / max_slf
        max_ns_pfa = max(y_ns_pfa_range) / max_slf

        # Target performance limit state
        y_target = [self.y[1], self.y[2]]
        y_s_psd = [y_target[0] * max_s_psd, y_target[1] * max_s_psd]
        y_ns_psd = [y_target[0] * max_ns_psd, y_target[1] * max_ns_psd]
        y_ns_pfa = [y_target[0] * max_ns_pfa, y_target[1] * max_ns_pfa]

        # Assuming NS and S have the same range
        interp_s_psd = interp1d(y_s_psd_range, psd_ns_range)
        interp_ns_psd = interp1d(y_ns_psd_range, psd_ns_range)
        interp_pfa = interp1d(y_ns_pfa_range, pfa_ns_range)
        # todo, the whole function needs an overhaul to adapt for each storey separately
        return y_target, [y_s_psd, y_ns_psd, y_ns_pfa], [interp_s_psd, interp_ns_psd, interp_pfa]
