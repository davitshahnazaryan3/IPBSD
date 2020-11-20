"""
user defines input arguments
Main file to run the software
"""
import pandas as pd
import numpy as np
from client.errorcheck import ErrorCheck
from client.hazard import Hazard
from client.spo import SPO
from external.spo2ida_call import spo2ida_allT


class Input:

    def __init__(self):
        """
        initializes the input functions
        """
        # TODO, separate definition of structure type into a different method
        # TODO, add dead structural weights to input loads as well?
        # input arguments
        self.i_d = None                             # Dictionary containing the original input arguments    dict
        self.case_id = None                         # Case ID for storing the data                          str
        self.PLS = None                             # Performance limit state names                         list(str)
        self.y = None                               # ELRs                                                  list(float)
        self.TR = None                              # Return periods in years                               list(float)
        self.beta_al = None                         # Uncertainties associated with PLSs                    list(float)
        self.nst = None                             # Number of storeys                                     int
        self.masses = None                          # Lumped masses at stories in tonne                     list(float)
        self.heights = None                         # Storey heights in m                                   list(float)
        self.o_th = None                            # Higher mode reduction factor                          float
        self.n_bays = None                          # Number of bays                                        int
        self.spans_x = None                         # Bay widths in m in X direction                        list(float)
        self.spans_y = None                         # Bay widths in m in Y direction                        list(float)
        self.fy = None                              # Steel yield strength in MPa                           float
        self.eps_y = None                           # Steel yield strain                                    float
        self.fc = None                              # Concrete compressive strength in MPa                  float
        self.n_seismic = None                       # Number of seismic frames                              int
        self.n_gravity = None                       # Number of gravity frames                              int
        self.bay_perp = None                        # Bay width (approximate) in the y direction            float
        self.w_seismic = None                       # Seismic weights in kN/m2                              dict
        self.pdelta_loads = None                    # Gravity loads over P Delta columns                    dict
        self.elastic_modulus_steel = 200000.        # Steel elastic modulus in MPa                          float

    def read_inputs(self, filename):
        """
        reads input data
        :param filename: str                        Filename containing input assumptions as path 'path/*.csv'
        :return: None
        """
        # Read the input file
        data = pd.read_csv(filename)
        # Check integrity of the input file
        self.i_d = {col: data[col].dropna().to_dict() for col in data}
        ErrorCheck(self.i_d)
        print("[SUCCESS] Integrity of input arguments are verified")
        self.case_id = self.i_d['design_scenario'][0]
        self.PLS = [self.i_d['PLS'][0], self.i_d['PLS'][1], self.i_d['PLS'][2]]
        self.y = [self.i_d['ELR'][0], self.i_d['ELR'][1], self.i_d['ELR'][2]]
        self.TR = [self.i_d['TR'][0], self.i_d['TR'][1], self.i_d['TR'][2]]
        self.beta_al = [self.i_d['aleatory'][0], self.i_d['aleatory'][1], self.i_d['aleatory'][2]]
        q_floor = self.i_d['bldg_ch'][0]
        q_roof = self.i_d['bldg_ch'][1]
        A_floor = self.i_d['bldg_ch'][2]
        self.heights = np.zeros(len(self.i_d['h_storeys']))
        for storey in range(len(self.i_d['h_storeys'])):
            self.heights[storey] = self.i_d['h_storeys'][storey]
        self.nst = len(self.heights)
        self.spans_x = []
        self.spans_y = []
        for bay in self.i_d['spans_X']:
            self.spans_x.append(self.i_d['spans_X'][bay])
        for bay in self.i_d['spans_Y']:
            self.spans_y.append(self.i_d['spans_Y'][bay])
        self.bay_perp = self.spans_y[0]
        self.n_bays = len(self.spans_x)
        # Loads and masses for the entire building (will be divided by n_seismic at later stages)
        self.masses = np.zeros(self.nst)
        self.pdelta_loads = np.zeros(self.nst)
        for storey in range(self.nst):
            if storey == self.nst - 1:
                self.masses[storey] = q_roof * A_floor / 9.81
                self.pdelta_loads[storey] = q_roof*(sum(self.spans_y)-self.bay_perp)
            else:
                self.masses[storey] = q_floor * A_floor / 9.81
                self.pdelta_loads[storey] = q_floor*(sum(self.spans_y)-self.bay_perp)
        self.o_th = self.i_d['mode_red'][0]
        self.fy = self.i_d['fy'][0]
        self.elastic_modulus_steel = self.i_d['Es'][0]
        self.eps_y = self.fy / self.elastic_modulus_steel
        self.fc = self.i_d['fc'][0]
        self.n_seismic = self.i_d['n_seismic_frames'][0]
        self.n_gravity = self.i_d['n_gravity_frames'][0]
        # Note: bay perpendicular used only for perimeter frames, which also assumes symmetry of the building
        # Consideration of space not included yet
        q_beam_floor = self.bay_perp / 2 * q_floor
        q_beam_roof = self.bay_perp / 2 * q_roof
        self.w_seismic = {'roof': q_beam_roof, 'floor': q_beam_floor}

    def read_hazard(self, flname, outputPath):
        """
        reads the provided seismic hazard function
        :param flname: str                              Hazard file name
        :param outputPath: str                          Outputs path
        :return: dicts                                  Fitted and original hazard information
        """
        h = Hazard(flname, outputPath)
        coefs, hazard_data, original_hazard = h.read_hazard()
        return coefs, hazard_data, original_hazard

    def initial_spo_data(self, period, filename):
        """
        spo parameters, initial assumption for the definition of the backbone curve
        :param period: float                            Assume a fundamental period of the structure
        :param filename: str                            Filename containing spo assumptions as path 'path/*.csv'
        :return: dict                                   Backbone curve parameters
        """
        # todo, future research, look into creating a database based on ML algorithms for possible SPO curve depending
        #  on the input
        data = pd.read_csv(filename)
        data = {col: data[col].dropna().to_dict() for col in data}
        spo = {'mc': data["mc"][0], 'a': data["a"][0], 'ac': data["ac"][0], 'r': data["r"][0], 'mf': data["mf"][0],
               'pw': data["pw"][0], 'T': period}
        return spo

    def read_spo(self, data, run_spo=False):
        """
        reads spo data
        :param data: decide on data type                Input data for running SPO2IDA tool
        :param run_spo: bool                            True if spo2ida needs to be run
        :return: floats, arrays                         84th, 50th, 16th percentiles of collapse capacity, IDAs
        """
        spo = SPO(data)
        mc = spo.mc
        a = spo.a
        ac = spo.ac
        r = spo.r
        mf = spo.mf
        pw = spo.pw
        period = spo.period
        if run_spo:
            R16, R50, R84, idacm, idacr, spom, spor = self.run_spo2ida(mc, a, ac, r, mf, period, pw)
            return R16, R50, R84, idacm, idacr, spom, spor

    def run_spo2ida(self, mc, a, ac, r, mf, period, pw):
        """
        runs spo2ida
        :param mc: float                                Hardening ductility
        :param a: float                                 Hardening slope
        :param ac: float                                Softening slope
        :param r: float                                 Residual strength, ratio of yield to residuals strength
        :param mf: float                                Fracturing ductility
        :param period: float                            First-mode period
        :param pw: float                                Pinching weight
        :return: floats, arrays                         84th, 50th, 16th percentiles of collapse capacity, IDAs
        """
        R16, R50, R84, idacm, idacr, spom, spor = spo2ida_allT(mc, a, ac, r, mf, period, pw)

        return R16, R50, R84, idacm, idacr, spom, spor
