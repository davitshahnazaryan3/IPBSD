"""
user defines input arguments
Main file to run the software
"""
from threading import Thread
import pandas as pd
import numpy as np


class Input:
    # Initialize input arguments
    inputs = None                       # Dictionary containing the original input arguments    dict
    case_id = None                      # Case ID for storing the data                          str
    PLS = None                          # Performance limit state names                         list(str)
    y = None                            # ELRs                                                  list(float)
    TR = None                           # Return periods in years                               list(float)
    beta_al = None                      # Uncertainties associated with PLSs                    list(float)
    nst = None                          # Number of storeys                                     int
    masses = None                       # Lumped masses at stories in tonne                     list(float)
    heights = None                      # Storey heights in m                                   list(float)
    o_th = None                         # Higher mode reduction factor                          float
    n_bays = None                       # Number of bays                                        int
    spans_x = None                      # Bay widths in m in X direction                        list(float)
    spans_y = None                      # Bay widths in m in Y direction                        list(float)
    fy = None                           # Steel yield strength in MPa                           float
    eps_y = None                        # Steel yield strain                                    float
    fc = None                           # Concrete compressive strength in MPa                  float
    n_seismic = None                    # Number of seismic frames                              int
    n_gravity = None                    # Number of gravity frames                              int
    bay_perp = None                     # Bay width (approximate) in the y direction            float
    w_seismic = None                    # Seismic weights in kN/m2                              dict
    pdelta_loads = None                 # Gravity loads over P Delta columns                    dict
    elastic_modulus_steel = 200000.     # Steel elastic modulus in MPa                          float
    configuration = None                # Space or Perimeter seismic frames                     str

    def __init__(self, flag3d=False):
        """
        initializes the input functions
        """
        self.flag3d = flag3d

    def read_inputs(self, filename):
        """
        reads input data
        :param filename: str                        Filename containing input assumptions as path 'path/*.csv'
        :return: None
        """
        # Read the input file
        data = pd.read_csv(filename)

        # Check integrity of the input file
        self.inputs = {col: data[col].dropna().to_dict() for col in data}
        # Case ID
        self.case_id = self.inputs['design_scenario'][0]

    def _get_performance_limit_states(self):
        self.PLS = [self.inputs['PLS'][0], self.inputs['PLS'][1], self.inputs['PLS'][2]]
        # Expected loss ratios
        self.y = [self.inputs['ELR'][0], self.inputs['ELR'][1], self.inputs['ELR'][2]]
        # Return periods
        self.TR = [int(self.inputs['TR'][0]), int(self.inputs['TR'][1]), int(self.inputs['TR'][2])]
        # Aleatory uncertainties
        self.beta_al = [self.inputs['aleatory'][0], self.inputs['aleatory'][1], self.inputs['aleatory'][2]]
        return {'PLS': self.PLS,
                'TR': self.TR,
                'beta_al': self.beta_al,
                'y': self.y}

    def _get_loads(self):
        loads = self.inputs['loads']
        return loads

    def _get_mode_reduction_factor(self):
        self.o_th = self.inputs['mode_red'][0]
        return self.o_th

    def _get_building_dimensions(self):
        # storey heights
        self.heights = np.zeros(len(self.inputs['heights']))
        for storey in range(len(self.inputs['heights'])):
            self.heights[storey] = self.inputs['heights'][storey]

        # bay widths in both principal directions
        self.spans_x = []
        self.spans_y = []
        for bay in self.inputs['spans_X']:
            self.spans_x.append(self.inputs['spans_X'][bay])
        for bay in self.inputs['spans_Y']:
            self.spans_y.append(self.inputs['spans_Y'][bay])

        # Configuration type for 3D modelling: space or perimeter
        self.configuration = self.inputs["configuration"][0]

        return self.heights, self.spans_x, self.spans_y, self.configuration

    def _get_material_props(self):
        # Reinforcement yield strength in MPa
        self.fy = self.inputs['fy'][0]
        # Reinforcement Elastic modulus in MPa
        self.elastic_modulus_steel = self.inputs['Es'][0]
        # Concrete compressive strength in MPa
        self.fc = self.inputs['fc'][0]
        # Reinforcement strain at yield
        self.eps_y = self.fy / self.elastic_modulus_steel
        return self.fc, self.fy, self.elastic_modulus_steel

    def get_input_arguments(self):
        # Get gravity (vertical) loads
        loads = self._get_loads()

        # General floor and roof loads in kPa
        q_floor = loads[0]
        q_roof = loads[1]

        # Number of storeys
        self.nst = len(self.heights)

        # Floor area in m2
        A_floor = sum(self.spans_x) * sum(self.spans_y)

        # Perpendicular bay width (important for 2D modelling)
        self.bay_perp = self.spans_y[0]
        # Number of bays along X
        self.n_bays = len(self.spans_x)

        # Loads and masses for the entire building (will be divided by n_seismic at later stages)
        self.masses = np.zeros(self.nst)
        # P-Delta loads for perimeter systems and 2D modelling
        self.pdelta_loads = np.zeros(self.nst)
        for storey in range(self.nst):
            if storey == self.nst - 1:
                self.masses[storey] = q_roof * A_floor / 9.81
                self.pdelta_loads[storey] = q_roof*(sum(self.spans_y)-self.bay_perp)*sum(self.spans_x)
            else:
                self.masses[storey] = q_floor * A_floor / 9.81
                self.pdelta_loads[storey] = q_floor*(sum(self.spans_y)-self.bay_perp)*sum(self.spans_x)

        # Important only for 2D modelling
        if self.configuration == "perimeter" or not self.flag3d:
            # Masses will be subdivided between two seismic frames
            self.n_seismic = 2
            self.n_gravity = int(len(self.spans_y) - 1)
        else:
            # Masses will be considered for the entirety of the building considering all seismic frames
            self.n_gravity = 0
            self.n_seismic = 1

        # Note: bay perpendicular used only for perimeter frames, which also assumes symmetry of the building
        # Consideration of space not included yet
        q_beam_floor = self.bay_perp / 2 * q_floor
        q_beam_roof = self.bay_perp / 2 * q_roof
        self.w_seismic = {'roof': q_beam_roof, 'floor': q_beam_floor}

    def run_all(self):
        Thread(target=self._get_performance_limit_states).start()
        Thread(target=self._get_mode_reduction_factor).start()
        Thread(target=self._get_building_dimensions).start()
        Thread(target=self._get_material_props).start()
