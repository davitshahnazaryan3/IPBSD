"""
identifies design limits for the verification of expected annual loss (EAL)
"""
import numpy as np
from client.slf import SLF


class DesignLimits:
    def __init__(self, slfDirectory, y, nst, flag3d=False, replCost=None):
        """
        Initialize SLF reading
        :param slfDirectory: str            Directory of SLFs derived via SLF Generator
        :param y: float                     Expected loss ratios (ELRs) associated with SLS
        :param nst: int                     Number of stories
        :param flag3d: bool                 False for "2d", True for "3d"
        :param replCost: float              Replacement cost of the entire building
        """
        self.slfDirectory = slfDirectory
        self.y = y
        self.nst = nst
        self.flag3d = flag3d
        self.theta_max = None               # Peak storey drift
        self.a_max = None                   # Peak floor acceleration in g
        self.SLFsCache = None
        self.replCost = replCost
        self.get_design_edps()
        
    def get_design_edps(self):
        """
        Calculates the design EDPs (i.e. PSD as theta and PFA as a)
        For a 3D building EDP limits will be calculated for each direction
        Non-directional SLFs and directional (corresponding to dir1 or dir2) will be summed
        """
        slf = SLF(self.slfDirectory, self.y, self.nst, self.flag3d, self.replCost)
        slfs, self.SLFsCache = slf.slfs()

        # Calculate the design limits of PSD and PFA beyond which EAL condition will not be met
        edp_limits = {}
        y_total = 0
        # Performance group
        for i in slfs["y"]:
            if i == "PFA_NS" or i == "PFA":
                group = "PFA"
            else:
                group = i

            edp_limits[group] = {}

            # Direction
            for k in slfs["y"][i]:
                edp_limits[group][k] = np.array([])

                # Story
                for st in slfs["y"][i][k]:
                    # ELRs
                    y = slfs["y"][i][k][st]
                    y_total += y
                    # SLF interpolation functions
                    s = slfs["interpolation"][i][k][st]
                    edp_limits[group][k] = np.append(edp_limits[group][k], float(s(y)))

        # Get the minimum value from structural and non-structural components
        edp_limits["PSD"] = {}
        for k in edp_limits["PSD_S"]:
            edp_limits["PSD"][k] = np.array([])
            for st in slfs["y"]["PSD_S"][k]:
                edp_limits["PSD"][k] = np.append(edp_limits["PSD"][k],
                                                 min(edp_limits["PSD_S"][k][int(st)-1],
                                                     edp_limits["PSD_NS"][k][int(st)-1]))

        # Design limits as min of the found values along the height
        if self.flag3d:
            n_dir = 2
        else:
            n_dir = 1
        self.theta_max = np.zeros(n_dir)
        self.a_max = np.zeros(n_dir)
        for i in range(len(self.theta_max)):
            self.theta_max[i] = round(min(edp_limits["PSD"][f"dir{i+1}"]), 4)
            self.a_max[i] = round(min(edp_limits["PFA"][f"dir{i+1}"]), 2)
