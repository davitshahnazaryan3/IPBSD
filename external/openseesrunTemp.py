"""
NOTE: This version includes nodes in the middle of the beams / which are not necessary, since in EFLM+gravity, the local
peaks are expected at the ends of the beams, rather than near the middle
Runs OpenSees software for structural analysis (in future may be changed with anastruct or other software)
For now only ELFM, pushover to be added later if necessary.
The idea is to keep it as simple as possible, and with fewer runs of this class.
This is  linear elastic analysis in order to obtain demands on structural components for further detailing
"""
import openseespy.opensees as ops
import pandas as pd
import numpy as np


class OpenSeesRun:
    def __init__(self, data, cs, analysis, fstiff=0.5, pflag=False):
        """
        initializes model creation for analysing
        :param data: dict                       Provided input arguments for the framework
        :param cs: DataFrame                    Cross-sections of the solution
        :param analysis: int                    Analysis type
        :param fstiff: float                    Stiffness reduction factor
        :param pflag: bool                      Print info
        """
        self.data = data
        self.cs = cs
        self.analysis = analysis
        self.fstiff = fstiff
        self.BEAM_TRANSF_TAG = 1
        self.COL_TRANSF_TAG = 2
        self.NEGLIGIBLE = 1.e-10
        self.pflag = pflag

    @staticmethod
    def wipe():
        """
        wipes model
        :return: None
        """
        ops.wipe()

    def create_model(self, mode='3D'):
        """
        creates the model
        :param mode: str                        2D or 3D
        :return: array                          Beam and column element tags
        """
        self.wipe()
        if mode == '3D':
            ops.model('Basic', '-ndm', 3, '-ndf', 6)
        elif mode == '2D':
            ops.model('Basic', '-ndm', 2, '-ndf', 3)
        else:
            print("[EXCEPTION] Wrong mode of analysis, select 2D or 3D!")
        self.define_nodes()
        self.define_transformations()
        beams, columns = self.create_elements()
        return beams, columns

    def define_transformations(self):
        """
        defines geometric transformations for beams and columns
        :return: None
        """
        col_transf_type = "PDelta"
        ops.geomTransf(col_transf_type, self.BEAM_TRANSF_TAG, 0, 1, 0)
        ops.geomTransf(col_transf_type, self.COL_TRANSF_TAG, 0, 1, 0)

    def define_nodes(self, fix_out_of_plane=True):
        """
        defines nodes and fixities
        :param fix_out_of_plane: bool           If 3D space is used for a 2D model then True
        :return: None
        """
        xloc = 0.
        for bay in range(0, int(self.data.n_bays + 1)):
            zloc = 0.
            # Base nodes, nodetag = f"{bay}{st}
            nodetag = int(str(1+bay)+'0')
            ops.node(nodetag, xloc, 0, zloc)
            ops.fix(nodetag, 1, 1, 1, 1, 1, 1)
            for st in range(1, self.data.nst + 1):
                zloc += self.data.h[st - 1]
                nodetag = int(str(1+bay) + str(st))
                # Nodes, y is taken as zero, if 2D frame is modelled in a 3D model space
                ops.node(nodetag, xloc, 0, zloc)
                # Create mid-beam nodes to capture mid-beam moments
                if bay != int(self.data.n_bays):
                    midX = xloc + self.data.i_d["spans_X"][bay] / 2
                    nodetag = int(str(1+bay) + str(st) + "00")
                    ops.node(nodetag, midX, 0, zloc)
            if bay == int(self.data.n_bays):
                pass
            else:
                xloc += self.data.spans_x[bay]
        # Fix out of plane modes, if 2D frame is modelled in a 3D model space
        if fix_out_of_plane:
            ops.fixY(0.0, 0, 1, 0, 0, 0, 1)
        if self.pflag:
            print("[NODE] Nodes created!")

    def lumped_hinge_element(self, et, gt, inode, jnode, my1, my2, lp, fc, b, h, ap=1.25, app=0.05,
                             r=0.1, mu_phi=10, pinch_x=0.8, pinch_y=0.5, damage1=0.0, damage2=0.0, beta=0.0,
                             mnode=None):
        # TODO, add other possible element models
        """
        creates a lumped hinge element
        :param et: int                          Element tag
        :param gt: int                          Geometric transformation tag
        :param inode: int                       i node (start node)
        :param jnode: int                       j node (end node)
        :param my1: float                       Yield moment at end i
        :param my2: float                       Yield moment at end j
        :param lp: float                        Plastic hinge length
        :param fc: float                        Concrete compressive strength
        :param b: float                         Element sectional width
        :param h: float                         Element sectional height
        :param ap: float                        Peak strength ratio
        :param app: float                       Post-peak strength ratio
        :param r: float                         Residual strength ratio
        :param mu_phi: float                    Curvature ductility
        :param pinch_x: float                   Pinching factor for strain (or deformation) during reloading
        :param pinch_y: float                   Pinching factor for stress (or force) during reloading
        :param damage1: float                   Damage due to ductility
        :param damage2: float                   Damage due to energy
        :param beta: float                      Power used to determine the degraded unloading stiffness based on
                                                ductility
        :param mnode: int                       Mid node, None for columns
        :return: None
        """
        elastic_modulus = (3320*np.sqrt(fc/1000) + 6900)*1000*self.fstiff
        area = b*h
        iz = b*h**3/12
        eiz = elastic_modulus*iz

        # Curvatures at yield
        phiy1 = my1/eiz
        phiy2 = my2/eiz

        # Material model creation
        hingeMTag1 = int(f"101{et}")
        hingeMTag2 = int(f"102{et}")
        mp1 = ap*my1
        mp2 = ap*my2
        mu1 = r*my1
        mu2 = r*my2

        phip1 = phiy1*mu_phi
        phip2 = phiy2*mu_phi
        phiu1 = phip1 + (mp1 - mu1)/(app*my1/phiy1)
        phiu2 = phip2 + (mp2 - mu2)/(app*my2/phiy2)

        ops.uniaxialMaterial("Hysteretic", hingeMTag1, my1, phiy1, mp1, phip1, mu1, phiu1,
                             -my1, -phiy1, -mp1, -phip1, -mu1, -phiu1, pinch_x, pinch_y, damage1, damage2, beta)
        ops.uniaxialMaterial("Hysteretic", hingeMTag2, my2, phiy2, mp2, phip2, mu2, phiu2,
                             -my2, -phiy2, -mp2, -phip2, -mu2, -phiu2, pinch_x, pinch_y, damage1, damage2, beta)

        # Element creation
        int_tag = int(f"105{et}")
        ph_tag1 = int(f"106{et}")
        ph_tag2 = int(f"107{et}")
        integration_tag = int(f"108{et}")

        ops.section("Elastic", int_tag, elastic_modulus, area, iz, iz, 0.4*elastic_modulus/self.fstiff, 0.01)
        ops.section("Uniaxial", ph_tag1, hingeMTag1, 'Mz')
        ops.section("Uniaxial", ph_tag2, hingeMTag2, 'Mz')

        ops.beamIntegration("HingeRadau", integration_tag, ph_tag1, lp, ph_tag2, lp, int_tag)

        if mnode is not None:
            # For beams
            # Create id for the second beam segment (starting with 3)
            et2 = int(200 + et)

            # Sub-element 1
            ops.element("forceBeamColumn", et, inode, mnode, gt, integration_tag)
            # Sub-element 2
            ops.element("forceBeamColumn", et2, mnode, jnode, gt, integration_tag)
        else:
            # For columns
            ops.element("forceBeamColumn", et, inode, jnode, gt, integration_tag)

    def create_elements(self):
        """
        creates elements
        consideration given only for ELFM, so capacities are arbitrary
        :return: list                               Element tags
        """
        n_beams = self.data.nst * self.data.n_bays
        n_cols = self.data.nst * (self.data.n_bays + 1)
        capacities_beam = [1000.0]*n_beams
        capacities_col = [1000.0]*n_cols

        # beam generation
        beam_id = 0
        b_beam = self.cs['b1']
        h_beam = self.cs['h1']
        lp = 1.0*h_beam         # not important for linear static analysis
        beams = []
        for bay in range(1, int(self.data.n_bays + 1)):
            for st in range(1, int(self.data.nst + 1)):
                next_bay = bay + 1
                my = capacities_beam[beam_id]
                beam_id += 1
                et = int(f"1{bay}{st}")
                gt = self.BEAM_TRANSF_TAG
                inode = int(f"{bay}{st}")
                jnode = int(f"{next_bay}{st}")
                mnode = int(inode*100)
                self.lumped_hinge_element(et, gt, inode, jnode, my, my, lp, self.data.fc, b_beam, h_beam, mnode=mnode)
                beams.append(et)
        if self.pflag:
            print("[ELEMENT] Beams created!")

        # column generation
        col_id = 0
        columns = []
        for bay in range(1, int(self.data.n_bays + 2)):
            for st in range(1, int(self.data.nst + 1)):
                previous_st = st - 1
                my = capacities_col[col_id]
                col_id += 1
                if bay == 1 or bay == 1 + int(self.data.n_bays):
                    b_col = h_col = self.cs[f'he{st}']
                else:
                    b_col = h_col = self.cs[f'hi{st}']
                lp = h_col*1.0         # not important for linear static analysis
                et = int(f"2{bay}{st}")
                gt = self.COL_TRANSF_TAG
                inode = int(f"{bay}{previous_st}")
                jnode = int(f"{bay}{st}")
                self.lumped_hinge_element(et, gt, inode, jnode, my, my, lp, self.data.fc, b_col, h_col)
                columns.append(et)
        if self.pflag:
            print("[ELEMENT] Columns created!")
        return beams, columns

    def gravity_loads(self, action, elements):
        # TODO, add possibility of point loads
        """
        Defines gravity loads
        :param action: list                         Acting gravity loads
        :param elements: list                       Element IDs
        :return: None
        """
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for ele in elements:
            storey = int(str(ele)[-1]) - 1
            ops.eleLoad('-ele', ele, '-type', '-beamUniform', action[storey], self.NEGLIGIBLE)

    def elfm_loads(self, action):
        """
        applies lateral loads
        :param action: list                         Acting lateral loads
        :return: None
        """
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for st in range(1, int(self.data.nst + 1)):
            for bay in range(1, int(self.data.n_bays + 2)):
                ops.load(int(f"{bay}{st}"), action[st-1] / (self.data.n_bays + 1), 0, 0, 0, 0, 0)

    def static_analysis(self):
        """
        carries out static ELFM analysis
        :return: None
        """
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-8, 6)
        ops.integrator("LoadControl", 0.1)
        ops.algorithm("Newton")
        ops.analysis("Static")
        ops.analyze(10)
        ops.loadConst("-time", 0.0)

    def define_recorders(self, beams, columns, analysis):
        """
        recording results
        :param beams: list                          Beam element tags
        :param columns: list                        Column element tags
        :param analysis: int                        Analysis type
        :return: ndarray                            Demands on beams and columns
        """
        if analysis != 4 and analysis != 5:
            b = np.zeros((self.data.nst, self.data.n_bays))
            c = np.zeros((self.data.nst, self.data.n_bays + 1))

            results = {"Beams": {"M": b.copy(), "N": b.copy(), "V": b.copy()},
                       "Columns": {"M": c.copy(), "N": c.copy(), "V": c.copy()}}

            # Beams
            ele = 0
            print(ops.eleForce(beams[ele]))
            for bay in range(self.data.n_bays):
                for st in range(self.data.nst):
                    results["Beams"]["M"][st][bay] = abs(max(ops.eleForce(beams[ele], 5),
                                                             ops.eleForce(beams[ele], 11), key=abs))
                    results["Beams"]["N"][st][bay] = abs(max(ops.eleForce(beams[ele], 1),
                                                             ops.eleForce(beams[ele], 7), key=abs))
                    results["Beams"]["V"][st][bay] = abs(max(ops.eleForce(beams[ele], 3),
                                                             ops.eleForce(beams[ele], 9), key=abs))
                    ele += 1

            # Columns
            ele = 0
            print(ops.eleForce(columns[ele]))
            for bay in range(self.data.n_bays + 1):
                for st in range(self.data.nst):
                    results["Columns"]["M"][st][bay] = abs(max(ops.eleForce(columns[ele], 5),
                                                               ops.eleForce(columns[ele], 11), key=abs))
                    results["Columns"]["N"][st][bay] = abs(max(ops.eleForce(columns[ele], 3),
                                                               ops.eleForce(columns[ele], 9), key=abs))
                    results["Columns"]["V"][st][bay] = abs(max(ops.eleForce(columns[ele], 1),
                                                               ops.eleForce(columns[ele], 7), key=abs))
                    ele += 1
        else:
            # TODO, fix recording when applying RMSA, analysis type if condition seems incorrect
            n_beams = self.data.nst * self.data.n_bays
            n_cols = self.data.nst * (self.data.n_bays + 1)
            results = {"Beams": {}, "Columns": {}}
            if analysis != 4 and analysis != 5:
                for i in range(n_beams):
                    results["Beams"][i] = {
                        "M": abs(max(ops.eleForce(beams[i], 5), ops.eleForce(beams[i], 11), key=abs)),
                        "N": abs(max(ops.eleForce(beams[i], 1), ops.eleForce(beams[i], 7), key=abs)),
                        "V": abs(max(ops.eleForce(beams[i], 3), ops.eleForce(beams[i], 9), key=abs))}
                for i in range(n_cols):
                    results["Columns"][i] = {
                        "M": abs(max(ops.eleForce(columns[i], 5), ops.eleForce(columns[i], 11), key=abs)),
                        "N": abs(max(ops.eleForce(columns[i], 3), ops.eleForce(columns[i], 9), key=abs)),
                        "V": abs(max(ops.eleForce(columns[i], 1), ops.eleForce(columns[i], 7), key=abs))}
            else:
                for i in range(n_beams):
                    results["Beams"][i] = {"M": np.array([ops.eleForce(beams[i], 5), ops.eleForce(beams[i], 11)]),
                                           "N": np.array([ops.eleForce(beams[i], 1), ops.eleForce(beams[i], 7)]),
                                           "V": np.array([ops.eleForce(beams[i], 3), ops.eleForce(beams[i], 9)])}
                for i in range(n_cols):
                    results["Columns"][i] = {"M": np.array([ops.eleForce(columns[i], 5), ops.eleForce(columns[i], 11)]),
                                             "N": np.array([ops.eleForce(columns[i], 3), ops.eleForce(columns[i], 9)]),
                                             "V": np.array([ops.eleForce(columns[i], 1), ops.eleForce(columns[i], 7)])}

        return results


if __name__ == "__main__":

    from client.master import Master
    from pathlib import Path
    directory = Path.cwd().parents[0]

    csd = Master(directory)
    csd.read_input("input.csv", "Hazard-LAquila-Soil-C.pkl")

    cs = {'he1': 0.35, 'hi1': 0.4, 'b1': 0.25, 'h1': 0.45, 'he2': 0.3, 'hi2': 0.35, 'b2': 0.25,
          'h2': 0.45, 'he3': 0.25, 'hi3': 0.3, 'b3': 0.25, 'h3': 0.45, 'T': 0.936}

    op = OpenSeesRun(csd.data, cs, analysis=3, fstiff=1.0)
    beams, columns = op.create_model()
    action = [160, 200, 200]
    gravity = [16.2, 13.5, 13.5]
    op.gravity_loads(gravity, beams)
    op.elfm_loads(action)
    op.static_analysis()
    response = op.define_recorders(beams, columns, analysis=1)
    # print(response)
