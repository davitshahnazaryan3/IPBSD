"""
identifies design limits for the verification of expected annual loss (EAL)
"""
import numpy as np
from client.slf import SLF


class DesignLimits:
    def __init__(self, slfDirectory, y, nst, geometry=0):
        """
        Initialize SLF reading
        :param slfDirectory: str            Directory of SLFs derived via SLF Generator
        :param y: float                     Expected loss ratios (ELRs) associated with SLS
        :param nst: int                     Number of stories
        :param geometry: int                0 for "2d", 1 for "3d"
        """
        self.slfDirectory = slfDirectory
        self.y = y
        self.nst = nst
        self.geometry = geometry
        self.theta_max = None               # Peak storey drift
        self.a_max = None                   # Peak floor acceleration in g
        self.SLFsCache = None

        self.get_design_edps()
        
    def get_design_edps(self):
        """
        Calculates the design EDPs (i.e. PSD as theta and PFA as a)
        """
        slf = SLF(self.slfDirectory, self.y, self.nst, self.geometry)
        slfs, self.SLFsCache = slf.slfs()

        # Calculate the design limits of PSD and PFA beyond which EAL condition will not be met
        edp_limits = {"PSD": np.array([]), "PFA": np.array([])}

        for i in slfs["y"]:
            if i == "PFA_NS" or i == "PFA":
                group = "PFA"
            else:
                group = "PSD"

            for st in slfs["y"][i]:
                # ELRs
                y = slfs["y"][i][st]
                # SLF interpolation functions
                s = slfs["interpolation"][i][st]

                edp_limits[group] = np.append(edp_limits[group], float(s(y)))

        # Design limits as min of the found values along the height
        self.theta_max = round(min(edp_limits["PSD"]), 4)
        self.a_max = round(min(edp_limits["PFA"]), 2)
