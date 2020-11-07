"""
performs design to spectral and vice-versa transformations
"""


class Transformations:
    def __init__(self, data, theta_max, a_max):
        """
        initialize
        :param data: class                          Processed input data
        :param theta_max: float                     PSD, [-]
        :param a_max: float                         PFA, [g]
        """
        self.data = data
        self.theta_max = theta_max
        self.a_max = a_max

    def table_generator(self):
        """
        Generates a table containing calculations at a limit state of interest following procedures similar to DBD
        :return: DataFrame, array, array            SLS table, 1st mode shape, storey displacements
        """
        table = {}
        nst = self.data.nst  # Number of storeys
        o_th = self.data.o_th  # Higher mode reduction factor
        h = self.data.h  # Storey heights of the structure
        Hn = sum(h)  # Total height of the structure
        deltas = []  # Initialization of lateral displacements
        for storey in range(nst):
            # Storey ID
            st_tag = str(storey + 1)
            # Initialize table
            table[st_tag] = {'H': 0., 'm': 0., 'delta': 0., 'theta': 0., 'phi': 0.}
            # Loop over each storey
            for s in range(storey + 1):
                # Storey height with respect to ground 0
                table[st_tag]['H'] = h[s] + table[st_tag]['H']
                # Masses lumped at each storey level for each seismic frame
                table[st_tag]['m'] = self.data.masses[storey] / self.data.n_seismic
            # Lateral displacements
            table[st_tag]['delta'] = o_th * self.theta_max * table[st_tag]['H'] * (4 * Hn - table[st_tag]['H']) / (
                        4 * Hn - h[0])
            if storey == 0:
                table[st_tag]['theta'] = table[st_tag]['delta'] / (table[st_tag]['H'])
            else:
                table[st_tag]['theta'] = (table[st_tag]['delta'] - table[str(storey)]['delta']) / \
                                         (table[st_tag]['H'] - table[str(storey)]['H'])
            deltas.append(table[st_tag]['delta'])

        # UPDATE --- Modal shape of RCF for now, assuming max Delta at top story
        # which is accurate for low to mid rise buildings
        phi = []
        for storey in range(nst):
            st_tag = str(storey + 1)
            table[st_tag]['phi'] = table[st_tag]['delta'] / max(deltas)
            phi.append(table[st_tag]['phi'])

        return table, phi, deltas

    def get_modal_parameters(self, phi):
        # TODO, does not seem like this method has a use
        """
        gets modal parameters
        :param phi: array                   1st mode shape
        :return: float, float               1st mode participation factor and effective mass
        """
        gamma = sum(map(lambda x, y: x * y, self.data.masses, phi)) / \
                sum(map(lambda x, y: x * y * y, self.data.masses, phi))
        mstar = sum(map(lambda x, y: x * y, self.data.masses, phi))

        return gamma, mstar

    def get_design_values(self, deltas):
        """
        gets the design spectral values corresponding to IDR and PFA
        :param deltas: array                Storey displacements
        :return: float, float               Design spectral displacement and acceleration
        """
        # Regression function for PFA conversion factor obtained from eigenvalue tests (needs improvement, models etc.)
        y_conv = (-int(self.data.nst) + 20.617) / 22.804

        delta_d = sum(map(lambda x, y: x * y * y, self.data.masses, deltas)) / \
                  sum(map(lambda x, y: x * y, self.data.masses, deltas))
        alpha_d = self.a_max * y_conv

        return delta_d, alpha_d
