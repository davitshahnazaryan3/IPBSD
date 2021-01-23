# Useful link: https://nptel.ac.in/courses/Webcourse-contents/IIT%20Kharagpur/Structural%20Analysis/pdf/m4l30.pdf

import numpy as np
import math
from scipy.linalg import eigh
from numpy.linalg import solve


class GetT1:
    def __init__(self, a_cols, a_c_ints, i_cols, i_c_ints, a_beams, i_beams, nst, spans_x, h, mi, n_seismic, fc,
                 fstiff=.5, just_period=False, w_seismic=None, mcy=None, mby=None, single_mode=True):
        """
        Initializes the package for the fundamental period of a frame
        :param a_cols: array                        Cross-section areas of external columns
        :param a_c_ints: array                      Cross-section areas of internal columns
        :param i_cols: array                        Moment of inertias of external columns
        :param i_c_ints: array                      Moment of inertias of internal columns
        :param a_beams: array                       Cross-section areas of beams
        :param i_beams: array                       Moment of inertias of beams
        :param nst: int                             Number of stories
        :param spans_x: array                       Bay widths
        :param h: array                             Storey heights
        :param mi: array                            Lumped storey masses
        :param n_seismic: int                       Number of seismic frames
        :param fc: float                            Concrete compressive strength
        :param fstiff: float                        Stiffness reduction factor (0.5 default)
        :param just_period: bool                    Check for period or also perform pushover
        :param w_seismic: array                     Seismic masses
        :param mcy: array                           Yield strength of columns
        :param mby: array                           Yield strength of beams
        :param single_mode: bool                    Whether to run only for 1st mode or multiple modes
                                                    1st mode only concerns the definition of solutions
                                                    Multiple modes are necessary when RSMA is performed at Action stage
        """
        self.a_cols = a_cols
        self.a_c_ints = a_c_ints
        self.i_cols = i_cols
        self.i_c_ints = i_c_ints
        self.a_beams = a_beams
        self.i_beams = i_beams
        self.nst = nst
        self.spans_x = spans_x
        self.h = h
        self.mi = mi
        self.n_seismic = n_seismic
        self.fc = fc
        self.fstiff = fstiff
        self.just_period = just_period
        self.w_seismic = w_seismic
        self.mcy = mcy
        self.mby = mby
        self.single_mode = single_mode

    def get_T_member(self, theta):
        """
        Get transformation matrix of element
        :param theta: float                         Element angle with respect to horizontal
        :return: array                              Transformation matrix
        """
        theta = np.radians(theta)
        l = np.cos(theta)
        m = np.sin(theta)
        t = [[l, m, 0, 0, 0, 0],
             [-m, l, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, l, m, 0],
             [0, 0, 0, -m, l, 0],
             [0, 0, 0, 0, 0, 1]]
        return np.array(t)

    def get_K_member(self, E, I, A, L, theta):
        """
        Get stiffness matrix of element
        :param E: float                             Elastic modulus of concrete
        :param I: float                             Moment of inertia of an element
        :param A: float                             Cross-section area of an element
        :param L: float                             Length of an element
        :param theta: float                         Element angle with respect to horizontal
        :return: array                              Global stiffness matrix
        """
        a = E * A / L
        b = 12 * E * I / L ** 3
        c = b / (2 / L)
        d = 4 * E * I / L
        e = d / 2
        theta = np.radians(theta)
        k_loc = [[a, 0, 0, -a, 0, 0],
                 [0, b, c, 0, -b, c],
                 [0, c, d, 0, -c, e],
                 [-a, 0, 0, a, 0, 0],
                 [0, -b, -c, 0, b, -c],
                 [0, c, e, 0, -c, d]]
        k_loc = np.array(k_loc)
        if theta != 0:
            l = np.cos(theta)
            m = np.sin(theta)
            T = [[l, m, 0, 0, 0, 0],
                 [-m, l, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, l, m, 0],
                 [0, 0, 0, -m, l, 0],
                 [0, 0, 0, 0, 0, 1]]
            T = np.array(T)
            temp = T.transpose().dot(k_loc)
            k_glob = temp.dot(T)
        else:
            k_glob = k_loc
        return k_glob

    def get_beam_fixed_fixed_reactions(self, w, L):
        """
        Gets beam fixed fixed reactions
        :param w: float                             Weight in kN/m
        :param L: float                             Length of an element
        :return: array                              Fixed-fixed reactions of a beam
        """
        a = 0
        b = w * L / 2
        c = w * L ** 2 / 12
        f = np.array([a, b, c, a, b, -c])
        return f

    def run_ma(self):
        """
        Runs modal analysis
        :return: float, list                        Return period and normalized 1st modal shape
        """
        nbays = len(self.spans_x)
        E = (3320 * np.sqrt(self.fc) + 6900) * 1000 * self.fstiff
        n_aligns_x = len(self.spans_x) + 1
        n_aligns_y = self.nst + 1
        n_nodes = n_aligns_x * n_aligns_y
        n_dofs = 3 * n_nodes
        k_frame = np.zeros((n_dofs, n_dofs))
        # Fill frame K for column matrices
        n_cols = n_aligns_x * (n_aligns_y - 1)
        theta_col = 90
        k_column_all = np.zeros(n_cols * 6 * 6).reshape(n_cols, 6, 6)
        # storey grouping
        for column in range(n_cols):
            column += 1
            if column in np.arange(1, n_cols - nbays + 1, nbays + 1) or \
                    column in np.arange(nbays + 1, n_cols + 1, nbays + 1):
                # External columns
                node_i = column
                idx_col = math.floor(node_i / n_aligns_x - 0.00001)
                l_col = self.h[idx_col]
                a_col = self.a_cols[idx_col]
                i_col = self.i_cols[idx_col]
            else:
                # Internal columns
                node_i = column
                idx_col = math.floor(node_i / n_aligns_x - 0.00001)
                l_col = self.h[idx_col]
                a_col = self.a_c_ints[idx_col]
                i_col = self.i_c_ints[idx_col]
            dofs_i = np.array([1, 2, 3]) + ((node_i - 1) * 3) - 1
            dofs_j = dofs_i + 3 * n_aligns_x
            dofs_i_i, dofs_i_j = dofs_i[0], dofs_i[-1] + 1
            dofs_j_i, dofs_j_j = dofs_j[0], dofs_j[-1] + 1

            k_column = self.get_K_member(E, i_col, a_col, l_col, theta_col)
            k_frame[dofs_i_i:dofs_i_j, :][:, dofs_i_i:dofs_i_j] += k_column[:3, :][:, :3]
            k_frame[dofs_i_i:dofs_i_j, :][:, dofs_j_i:dofs_j_j] += k_column[:3, :][:, 3:]
            k_frame[dofs_j_i:dofs_j_j, :][:, dofs_i_i:dofs_i_j] += k_column[3:, :][:, :3]
            k_frame[dofs_j_i:dofs_j_j, :][:, dofs_j_i:dofs_j_j] += k_column[3:, :][:, 3:]
            k_column_all[column - 1] = k_column

        # Fill frame K for beam matrices
        n_beams = (n_aligns_x - 1) * (n_aligns_y - 1)
        theta_beam = 0
        f_grav_frame = np.zeros(n_dofs)
        k_beam_all = np.zeros(n_beams * 6 * 6).reshape(n_beams, 6, 6)
        count_bay = 0
        count_st = 0
        for beam in range(n_beams):
            beam += 1
            floor = math.floor((beam - 1) / (n_aligns_x - 1))
            node_i = beam + n_aligns_x + floor
            # Indices for beam parameters
            idx_beam_bay = count_bay
            idx_beam_st = count_st
            l_beam = self.spans_x[idx_beam_bay]
            a_beam = self.a_beams[idx_beam_bay][idx_beam_st]
            i_beam = self.i_beams[idx_beam_bay][idx_beam_st]
            dofs_i = np.array([1, 2, 3]) + ((node_i - 1) * 3) - 1
            dofs_j = dofs_i + 3
            dofs_i_i, dofs_i_j = dofs_i[0], dofs_i[-1] + 1
            dofs_j_i, dofs_j_j = dofs_j[0], dofs_j[-1] + 1
            k_beam = self.get_K_member(E, i_beam, a_beam, l_beam, theta_beam)
            k_frame[dofs_i_i:dofs_i_j, :][:, dofs_i_i:dofs_i_j] += k_beam[:3, :][:, :3]
            k_frame[dofs_i_i:dofs_i_j, :][:, dofs_j_i:dofs_j_j] += k_beam[:3, :][:, 3:]
            k_frame[dofs_j_i:dofs_j_j, :][:, dofs_i_i:dofs_i_j] += k_beam[3:, :][:, :3]
            k_frame[dofs_j_i:dofs_j_j, :][:, dofs_j_i:dofs_j_j] += k_beam[3:, :][:, 3:]
            k_beam_all[beam - 1] = k_beam
            if not self.just_period:
                if floor != nst - 1:
                    w_member = self.w_seismic['floor']
                else:
                    w_member = self.w_seismic['roof']
                f_member = self.get_beam_fixed_fixed_reactions(w_member, l_beam)
                f_grav_frame[dofs_i_i:dofs_i_j] += f_member[:3]
                f_grav_frame[dofs_j_i:dofs_j_j] += f_member[3:]
            count_bay += 1
            if count_bay == n_aligns_x - 1:
                count_bay = 0
            if count_bay == 0:
                count_st += 1

        # Remove rows/columns of supports
        dof_start = n_aligns_x * 3
        k_frame = k_frame[dof_start:, :][:, dof_start:]
        f_grav_frame = f_grav_frame[dof_start:]
        # Assemble mass matrix
        mi_frame = self.mi / self.n_seismic
        mi_node = mi_frame / n_aligns_x
        mi_diag = np.repeat(mi_node, n_aligns_x * 3)
        mi_diag[1::3] = 1e-5
        mi_diag[2::3] = 1e-5
        m_frame = np.diag(mi_diag)
        # Calculate T1
        # Mode 1: eigvals=(0,0), Period [0][0], Phis [1][x]
        T = 2 * np.pi / (eigh(k_frame, m_frame, eigvals=(0, 0))[0][0] ** 0.5)
        phis = np.zeros(self.nst)
        phi_norm = np.zeros((self.nst, 1))

        if self.single_mode:
            for storey in range(self.nst):
                phis[storey] = abs(eigh(k_frame, m_frame, eigvals=(0, 0))[1][storey * (nbays * 3 + 3)])
            for i in range(len(phis)):
                phi_norm[i] = phis[i] / max(phis)
        else:
            T = np.zeros(self.nst)
            n_modes = self.nst
            phi_all = np.zeros((n_modes, self.nst))
            phi_all_norm = np.zeros((n_modes, self.nst))
            for j in range(n_modes):
                T[j] = 2*np.pi/(eigh(k_frame, m_frame, eigvals=(j, j))[0][0]**.5)
            for storey in range(self.nst):
                phis[storey] = -(eigh(k_frame, m_frame, eigvals=(0, 0))[1][storey*(nbays*3+3)])
            for i in range(len(phis)):
                phi_norm[i] = phis[i] / max(abs(phis))
            for j in range(n_modes):
                for st in range(self.nst):
                    if j == 0:
                        phi_all[j, st] = -(eigh(k_frame, m_frame, eigvals=(j, j))[1][st * (nbays * 3 + 3)])
                    else:
                        phi_all[j, st] = (eigh(k_frame, m_frame, eigvals=(j, j))[1][st * (nbays * 3 + 3)])
            for j in range(n_modes):
                for st in range(self.nst):
                    phi_all_norm[j, st] = phi_all[j, st] / max(abs(phi_all[j, :]))
            phi_norm = phi_all_norm

        if self.just_period:
            return T, phi_norm

        else:
            """ If procedure is being called for in seismic context, do "pushover"
            analysis to estimate the yield displacement of the solution
            "Pushover" determines the load at which 1st yielding occurs
            Very important to include gravity bending moments to avoid the same
            problem of overstrength in EC8
            By knowing the pushover load factor at which 1st yielding occurs, we
            can know the yield displacement of the frame.
            """
            # todo, add this portion, clean
            # Calculate pushover load pattern for a 1kN base shear
            zi_node = np.cumsum(h)
            zi_diag = np.repeat(zi_node, n_aligns_x * 3)
            zi_diag[1::3] = 1e-5
            zi_diag[2::3] = 1e-5
            Fi = mi_diag * zi_diag / (mi_diag * zi_diag).sum()
            # Calculate nodal displacements of the frame due to pushover load
            u_push = solve(k_frame, Fi)
            # Calculate nodal displacements of the frame due to gravity loads
            u_grav = solve(k_frame, f_grav_frame)
            # For each beam, get pushover scaling factor that leads to yielding
            lbd_beam_all = np.zeros((n_beams, 2))
            for beam in range(n_beams):
                beam += 1
                floor = math.floor((beam - 1) / (n_aligns_x - 1))
                idx_beam_bay = floor
                node_i = beam + n_aligns_x + floor
                My = self.mby[idx_beam_bay]
                dofs_i = np.array([1, 2, 3]) + ((node_i - 1) * 3) - 1
                dofs_j = dofs_i + 3
                dofs_interest = list(dofs_i - dof_start) + list(dofs_j - dof_start)
                u_member_push = u_push[dofs_interest]
                u_member_grav = u_grav[dofs_interest]
                k_member_glob = k_beam_all[beam - 1]
                f_member_push_glob = k_member_glob.dot(u_member_push)
                f_member_grav_glob = k_member_glob.dot(u_member_grav)
                t_member = self.get_T_member(theta_beam)
                f_member_push_loc = t_member.dot(f_member_push_glob)
                f_member_grav_loc = t_member.dot(f_member_grav_glob)
                lbd_beam_all[beam - 1][0] = (My - abs(f_member_grav_loc[0, 2])) / abs(f_member_push_loc[0, 2])
                lbd_beam_all[beam - 1][1] = (My - abs(f_member_grav_loc[0, 5])) / abs(f_member_push_loc[0, 5])
            # For each column, get pushover scaling factor that leads to yielding
            lbd_col_all = np.zeros((n_cols, 2))
            for column in range(1):
                column += 1
                node_i = column
                idx_col = math.floor(node_i / (n_aligns_x + 1))
                My = self.mcy[idx_col]
                dofs_i = np.array([1, 2, 3]) + ((node_i - 1) * 3) - 1
                dofs_j = dofs_i + 3 * n_aligns_x
                dofs_interest = list(dofs_i - dof_start) + list(dofs_j - dof_start)
                u_member_push = u_push[dofs_interest]
                u_member_grav = u_grav[dofs_interest]
                u_member_push[np.array(dofs_interest) < 0] = 0
                u_member_grav[np.array(dofs_interest) < 0] = 0
                k_member_glob = k_column_all[column - 1]
                f_member_push_glob = k_member_glob.dot(u_member_push)
                f_member_grav_glob = k_member_glob.dot(u_member_grav)
                t_member = self.get_T_member(theta_col)
                f_member_push_loc = t_member.dot(f_member_push_glob)
                f_member_grav_loc = t_member.dot(f_member_grav_glob)
                lbd_col_all[column - 1][0] = (My - abs(f_member_grav_loc[0, 2])) / abs(f_member_push_loc[0, 2])
                lbd_col_all[column - 1][1] = (My - abs(f_member_grav_loc[0, 5])) / abs(f_member_push_loc[0, 5])
            # Create flags for potential soft-storey mechanisms at 1st yielding
            lbd_beam_min = lbd_beam_all.min()
            lbd_col_min = lbd_col_all.min()
            lbd_min = min(lbd_beam_min, lbd_col_min)
            rip_frame = 0
            # If first yielding does not occur in beams
            if lbd_min == lbd_col_min:
                test_cols = lbd_col_all == lbd_col_min
                # If first yielding occurs in any column other than 1st storey.
                # This could also be modified to allow yielding at uppermost storey
                test = test_cols[n_aligns_x:, :].sum()
                if test != 0:
                    rip_frame = 1
                # If first yielding occurs in 1st storey, but not at the base.
                test = test_cols[:n_aligns_x, 1].sum()
                if test != 0:
                    rip_frame = 1
            # Weak-beam-strong-column is not being satisfied

            # Calculate yield base shear and yield displacement
            #        - calculate Gamma of the frame (need mode shape, masses)
            #        Vb1y=Fi.sum()*lbd_min
            #        SaSDOF=(Vb1y/Mi_total)/Gamma

            return T


if __name__ == "__main__":

    nst = 3
    nbays = 2

    b_col = [.55, .55, .55]
    h_col = b_col
    b_beam = np.array([[.55, .5, .45], [.55, .5, .45]])
    h_beam = np.array([[.7, .65, .6], [.7, .65, .6]])
    spans_X = [6., 6.5]
    h = [3.5, 3., 3.]
    mi = np.array([128.*2, 121.*2, 121.*2])
    n_seismic = 2
    fc = 25.
    fstiff = 0.5
    w_seismic = {'roof': 0, 'floor': 0}  # seismic loads do not impact the results for now

    A_cols = []
    I_cols = []
    A_beams = b_beam * h_beam
    I_beams = b_beam * h_beam ** 3 / 12
    A_c_ints = []
    I_c_ints = []
    for i in range(nst):
        A_cols.append(b_col[i] * h_col[i])
        I_cols.append(b_col[i] * h_col[i] ** 3 / 12)
        A_c_ints.append(0.7 * 0.7)
        I_c_ints.append(0.7 * 0.7 ** 3 / 12)
    gt = GetT1(A_cols, A_c_ints, I_cols, I_c_ints, A_beams, I_beams, nst, spans_X, h, mi, n_seismic, fc, fstiff,
               just_period=True, w_seismic=w_seismic)
    T1, phi_norm = gt.run_ma()
    print('Fundamental period is: {}'.format(T1))
