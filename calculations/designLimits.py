"""
identifies design limits for the verification of expected annual loss (EAL)
"""
import numpy as np
from client.slf import SLF


class DesignLimits:
    def __init__(self, slfDirectory, y, nst, flag3d=False, replCost=None, eal_corrections=True, perform_scaling=True,
                 edp_profiles=None):
        """
        Initialize SLF reading
        :param slfDirectory: str            Directory of SLFs derived via SLF Generator
        :param y: float                     Expected loss ratios (ELRs) associated with SLS
        :param nst: int                     Number of stories
        :param flag3d: bool                 False for "2d", True for "3d"
        :param replCost: float              Replacement cost of the entire building
        :param eal_corrections: bool        Perform EAL corrections
        :param perform_scaling: bool        Perform scaling of SLFs to replCost
        """
        self.slfDirectory = slfDirectory
        self.y = y
        self.nst = nst
        self.flag3d = flag3d
        self.theta_max = None               # Peak storey drift
        self.a_max = None                   # Peak floor acceleration in g
        self.SLFsCache = None
        self.contributions = None
        self.replCost = replCost
        self.eal_corrections = eal_corrections
        self.perform_scaling = perform_scaling
        self.edp_profiles = edp_profiles
        self.get_design_edps()
        
    def get_design_edps(self):
        """
        Calculates the design EDPs (i.e. PSD as theta and PFA as a)
        For a 3D building EDP limits will be calculated for each direction
        Non-directional SLFs and directional (corresponding to dir1 or dir2) will be summed
        """
        slf = SLF(self.slfDirectory, self.y, self.nst, self.flag3d, self.replCost, self.perform_scaling)
        slfs, self.SLFsCache = slf.slfs()

        # Calculate the design limits of PSD and PFA beyond which EAL condition will not be met
        edp_limits = {}
        # Initialize contributions
        self.contributions = {"y_PSD_S": [], "y_PSD_NS": [], "y_PFA_NS": [], "PSD_S": [], "PSD_NS": [], "PFA_NS": []}

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

                # Storey
                for st in slfs["y"][i][k]:
                    # ELRs
                    y = slfs["y"][i][k][st]
                    # SLF interpolation functions
                    s = slfs["interpolation"][i][k][st]
                    edp_limits[group][k] = np.append(edp_limits[group][k], float(s(y)))
                    if group == "PFA" and k == "dir2":
                        pass
                    else:
                        self.contributions[i].append(float(s(y)))
                        self.contributions["y_" + i].append(y)

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
            self.theta_max[i] = round(min(edp_limits["PSD"][f"dir{i+1}"]), 5)
            self.a_max[i] = round(min(edp_limits["PFA"][f"dir{i+1}"]), 3)

        # If EAL corrections need to be performed, the ELRs and EAL need to be recalculated
        if self.eal_corrections:
            y_perf_group = {}
            y_total_slf = np.zeros((2, ))
            y_total_sls = np.zeros((2, ))

            # Initialize contributions
            self.contributions = {"y_PSD_S": [], "y_PSD_NS": [], "y_PFA_NS": [], "PSD_S": [], "PSD_NS": [], "PFA_NS": []}

            # Performance group
            for i in slfs["y"]:
                if i == "PFA_NS" or i == "PFA":
                    group = "PFA"
                else:
                    group = i

                y_perf_group[group] = {}

                # Direction
                for k in slfs["y"][i]:
                    y_perf_group[group][k] = np.array([])
                    if group == "PFA":
                        edp = self.a_max[int(k[-1]) - 1]
                    else:
                        edp = self.theta_max[int(k[-1]) - 1]
                    # Storey
                    for st in slfs["y"][i][k]:
                        # Use specific profiles if provided
                        if self.edp_profiles is not None:
                            if group == "PFA":
                                edp = self.a_max[int(k[-1]) - 1] * self.edp_profiles[0][int(st)]
                            else:
                                edp = self.theta_max[int(k[-1]) - 1] * self.edp_profiles[1][int(st)-1]
                        if group == "PFA" and k == "dir2":
                            # To avoid double counting PFA sensitive components
                            pass
                        else:
                            y_total_slf[int(k[-1]) - 1] += slfs["y"][i][k][st] / self.y

                            # SLF interpolation functions
                            s = slfs["edp_interpolation"][i][k][st]
                            y_perf_group[group][k] = np.append(y_perf_group[group][k], float(s(edp)))
                            y_total_sls[int(k[-1]) - 1] += float(s(edp))
                            self.contributions[i].append(edp)
                            self.contributions["y_" + i].append(float(s(edp)))

            # Recalculate ELR at SLS
            self.y = sum(y_total_sls[:n_dir]) / sum(y_total_slf[:n_dir])
