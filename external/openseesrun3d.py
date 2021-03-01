"""
Runs OpenSees software for structural analysis (in future may be changed with anastruct or other software)
For now only ELFM, pushover to be added later if necessary.
The idea is to keep it as simple as possible, and with fewer runs of this class.
This is  linear elastic analysis in order to obtain demands on structural components for further detailing
"""
import openseespy.opensees as op
import numpy as np


class OpenSeesRun3D:
    def __init__(self, i_d, cs, fstiff=0.5, hinge=None, pflag=False, direction=0):
        """
        initializes model creation for analysing
        :param i_d: dict                        Provided input arguments for the framework
        :param cs: dict                         DataFrames of Cross-sections of the solution
        :param fstiff: float                    Stiffness reduction factor
        :param hinge: dict                      DataFrames of Idealized plastic hinge model parameters
        :param pflag: bool                      Print info
        :param direction: bool                  0 for x direction, 1 for y direction
        """
        self.i_d = i_d
        self.cs = cs
        self.fstiff = fstiff
        self.NEGLIGIBLE = 1.e-10
        self.UBIG = 1.e10
        self.pflag = pflag
        self.hinge = hinge
        self.direction = direction
        # Base nodes for SPO recorder
        self.base_nodes = []
        # Base columns (base node reaction recorders seem to be not working)
        self.base_cols = []
        # Rightmost nodes for pushover load application
        self.spo_nodes = []
        # Transformation tags
        self.COL_TRANSF_TAG = 1
        self.BEAM_X_TRANSF_TAG = 2
        self.BEAM_Y_TRANSF_TAG = 3
        # Yield moment constant, placeholder
        self.MY_CONSTANT = 10.

    @staticmethod
    def wipe():
        """
        wipes model
        :return: None
        """
        op.wipe()

    def get_elastic_modulus(self, hinge=None, st=None, element=None, inertia=None, bay=None):
        if hinge is not None:
            if bay is not None:
                ele_hinge = hinge[(hinge["Element"] == element) & (hinge["Bay"] == bay) & (
                        hinge["Storey"] == st)].reset_index(drop=True)
            else:
                ele_hinge = hinge[(hinge["Element"] == element) & (hinge["Storey"] == st)].reset_index(drop=True)

            my = ele_hinge["m1"].iloc[0]
            phiy = ele_hinge["phi1"].iloc[0]
            el_mod = my / phiy / inertia

        else:
            el_mod = (3320 * np.sqrt(self.i_d.fc) + 6900) * 1000 * self.fstiff

        return el_mod

    def static_analysis(self):
        op.constraints("Penalty", 1.0e15, 1.0e15)
        op.numberer("RCM")
        op.system("UmfPack")
        op.test("EnergyIncr", 1.0e-8, 6)
        op.integrator("LoadControl", 1.0)
        op.algorithm("Linear")
        op.analysis("Static")
        op.analyze(1)
        op.loadConst("-time", 0.0)

    def recorders(self, nbays_x, nbays_y, direction=0):

        # Indices for recorders
        if direction == 0:
            # Along X axis
            midx = [11, 5]
            vidx_c = [1, 7]
            vidx_b = [3, 9]
            nidx_c = [3, 9]
            nidx_b = [1, 7]
        else:
            # Along Y axis
            midx = [10, 4]
            vidx_c = [2, 8]
            vidx_b = [3, 9]
            nidx_c = [3, 9]
            nidx_b = [2, 8]

        # Define recorders
        # Seismic frame element recorders initialization
        bx_seismic = np.zeros((self.i_d.nst, nbays_x))
        by_seismic = np.zeros((self.i_d.nst, nbays_y))
        cx_seismic = np.zeros((self.i_d.nst, nbays_x + 1))
        cy_seismic = np.zeros((self.i_d.nst, nbays_y + 1))

        # For gravity frame elements only the max demands will be used for uniform design
        bx_gravity = np.zeros((self.i_d.nst, nbays_x, nbays_y - 1))
        by_gravity = np.zeros((self.i_d.nst, nbays_x - 1, nbays_y))
        c_gravity = np.zeros((self.i_d.nst, nbays_x - 1, nbays_y - 1))

        # Global dictionary variable to store all demands
        results = {"x_seismic": {"Beams": {"M": {"Pos": bx_seismic.copy(), "Neg": bx_seismic.copy()},
                                           "N": bx_seismic.copy(), "V": bx_seismic.copy()},
                                 "Columns": {"M": cx_seismic.copy(), "N": cx_seismic.copy(), "V": cx_seismic.copy()}},
                   "y_seismic": {"Beams": {"M": {"Pos": by_seismic.copy(), "Neg": by_seismic.copy()},
                                           "N": by_seismic.copy(), "V": by_seismic.copy()},
                                 "Columns": {"M": cy_seismic.copy(), "N": cy_seismic.copy(), "V": cy_seismic.copy()}},
                   "gravity": {"Beams_x": {"M": {"Pos": bx_gravity.copy(), "Neg": bx_gravity.copy()},
                                           "N": bx_gravity.copy(), "V": bx_gravity.copy()},
                               "Beams_y": {"M": {"Pos": by_gravity.copy(), "Neg": by_gravity.copy()},
                                           "N": by_gravity.copy(), "V": by_gravity.copy()},
                               "Columns": {"M": c_gravity.copy(), "N": c_gravity.copy(), "V": c_gravity.copy()}}}

        # Beams, counting: bottom to top, left to right
        # Beams of seismic frame along X direction
        # Beam iNode [Fx, Fy, Fz, Mx, My, Mz]; jNode [Fx, Fy, Fz, Myx, My, Mz]
        # iNode Negative My means upper demand; jNode Positive My means upper demand
        for bay in range(nbays_x):
            for st in range(self.i_d.nst):
                et = int(f"3{bay+1}1{st+1}")
                results["x_seismic"]["Beams"]["M"]["Pos"][st][bay] = abs(op.eleForce(et, midx[1]))
                results["x_seismic"]["Beams"]["M"]["Neg"][st][bay] = abs(op.eleForce(et, midx[0]))
                results["x_seismic"]["Beams"]["N"][st][bay] = max(op.eleForce(et, nidx_b[0]),
                                                                  op.eleForce(et, nidx_b[1]), key=abs)
                results["x_seismic"]["Beams"]["V"][st][bay] = max(abs(op.eleForce(et, vidx_b[0])),
                                                                  abs(op.eleForce(et, vidx_b[1])))

        # Beams of seismic frame along Y direction
        for bay in range(nbays_y):
            for st in range(self.i_d.nst):
                et = int(f"21{bay+1}{st+1}")
                results["y_seismic"]["Beams"]["M"]["Pos"][st][bay] = abs(op.eleForce(et, midx[1]))
                results["y_seismic"]["Beams"]["M"]["Neg"][st][bay] = abs(op.eleForce(et, midx[0]))
                results["y_seismic"]["Beams"]["N"][st][bay] = max(op.eleForce(et, nidx_b[0]),
                                                                  op.eleForce(et, nidx_b[1]), key=abs)
                results["y_seismic"]["Beams"]["V"][st][bay] = max(abs(op.eleForce(et, vidx_b[0])),
                                                                  abs(op.eleForce(et, vidx_b[1])))

        # Beams of gravity frames along X direction
        for ybay in range(nbays_y - 1):
            for xbay in range(nbays_x):
                for st in range(self.i_d.nst):
                    et = int(f"3{xbay+1}{ybay+2}{st+1}")
                    results["gravity"]["Beams_x"]["M"]["Pos"][st][xbay][ybay] = abs(op.eleForce(et, midx[1]))
                    results["gravity"]["Beams_x"]["M"]["Neg"][st][xbay][ybay] = abs(op.eleForce(et, midx[0]))
                    results["gravity"]["Beams_x"]["N"][st][xbay][ybay] = max(op.eleForce(et, nidx_b[0]),
                                                                             op.eleForce(et, nidx_b[1]), key=abs)
                    results["gravity"]["Beams_x"]["V"][st][xbay][ybay] = max(abs(op.eleForce(et, 3)),
                                                                             abs(op.eleForce(et, 9)))

        # Beams of gravity frames along Y direction
        for xbay in range(nbays_x - 1):
            for ybay in range(nbays_y):
                for st in range(self.i_d.nst):
                    et = int(f"2{xbay + 2}{ybay + 1}{st + 1}")
                    results["gravity"]["Beams_y"]["M"]["Pos"][st][xbay][ybay] = abs(op.eleForce(et, midx[1]))
                    results["gravity"]["Beams_y"]["M"]["Neg"][st][xbay][ybay] = abs(op.eleForce(et, midx[0]))
                    results["gravity"]["Beams_y"]["N"][st][xbay][ybay] = max(op.eleForce(et, nidx_b[0]),
                                                                             op.eleForce(et, nidx_b[1]), key=abs)
                    results["gravity"]["Beams_y"]["V"][st][xbay][ybay] = max(abs(op.eleForce(et, vidx_b[0])),
                                                                             abs(op.eleForce(et, vidx_b[1])))

        # Columns
        # Columns of seismic frame along x direction
        # Columns [Vx, Vy, N, Mx, My, Mz]; jNode [Vx, Vy, N, Mx, My, Mz]
        # Columns for X direction demand estimations [V, 2, N, 3, M, 6]
        # Columns for Y direction demand estimations [0, V, N, M, 5, 6]
        for bay in range(nbays_x + 1):
            for st in range(self.i_d.nst):
                et = int(f"1{bay+1}1{st+1}")
                results["x_seismic"]["Columns"]["M"][st][bay] = max(abs(op.eleForce(et, midx[0])),
                                                                    abs(op.eleForce(et, midx[1])))
                # Negative N is tension, Positive N is compression
                results["x_seismic"]["Columns"]["N"][st][bay] = max(op.eleForce(et, nidx_c[0]),
                                                                    op.eleForce(et, nidx_c[1]), key=abs)
                results["x_seismic"]["Columns"]["V"][st][bay] = max(abs(op.eleForce(et, vidx_c[0])),
                                                                    abs(op.eleForce(et, vidx_c[1])))

        # Columns of seismic frame along y direction
        for bay in range(nbays_y + 1):
            for st in range(self.i_d.nst):
                et = int(f"11{bay+1}{st+1}")
                results["y_seismic"]["Columns"]["M"][st][bay] = max(abs(op.eleForce(et, midx[0])),
                                                                    abs(op.eleForce(et, midx[1])))
                # Negative N is tension, Positive N is compression
                results["y_seismic"]["Columns"]["N"][st][bay] = max(op.eleForce(et, nidx_c[0]),
                                                                    op.eleForce(et, nidx_c[1]), key=abs)
                results["y_seismic"]["Columns"]["V"][st][bay] = max(abs(op.eleForce(et, vidx_c[0])),
                                                                    abs(op.eleForce(et, vidx_c[1])))

        # Columns of gravity frames
        for xbay in range(nbays_x - 1):
            for ybay in range(nbays_y - 1):
                for st in range(self.i_d.nst):
                    et = int(f"1{xbay+2}{ybay+2}{st+1}")
                    results["gravity"]["Columns"]["M"][st][xbay][ybay] = max(abs(op.eleForce(et, midx[0])),
                                                                             abs(op.eleForce(et, midx[1])))
                    # Negative N is tension, Positive N is compression
                    results["gravity"]["Columns"]["N"][st][xbay][ybay] = max(op.eleForce(et, nidx_c[0]),
                                                                             op.eleForce(et, nidx_c[1]), key=abs)
                    results["gravity"]["Columns"]["V"][st][xbay][ybay] = max(abs(op.eleForce(et, vidx_c[0])),
                                                                             abs(op.eleForce(et, vidx_c[1])))

        # Recording Base node reactions
        op.reactions()
        for xbay in range(1, nbays_x + 2):
            for ybay in range(1, nbays_y + 2):
                nodetag = int(f"{xbay}{ybay}0")
                # print(op.nodeReaction(nodetag))

        return results

    def apply_gravity_loads(self, beams, q_floor, q_roof, spans_x, spans_y):
        for d in beams:
            for beam in beams[d]:
                st = int(str(beam)[-1])
                xbay = int(str(beam)[1])
                ybay = int(str(beam)[2])

                # Distributed loads (kN/m2)
                if st != self.i_d.nst:
                    # General storey
                    q = q_floor
                else:
                    # Roof level
                    q = q_roof

                # Distributing the loads
                if d == "x" or d == "gravity_x":
                    # Beams along X direction
                    # Load over a beam
                    control_length = spans_y[ybay - 1] if ybay < len(spans_y) + 1 else spans_y[ybay - 2]
                    if spans_x[xbay - 1] <= control_length:
                        # Triangular rule
                        load = q * spans_x[xbay - 1] ** 2 / 4 / spans_x[xbay - 1]
                    else:
                        # Trapezoidal rule
                        load = 1 / 4 * q * spans_y[ybay - 1] * (2 * spans_x[xbay - 1] - spans_y[ybay - 1]) / \
                               spans_x[xbay - 1]

                    op.eleLoad('-ele', beam, '-type', '-beamUniform', -load, self.NEGLIGIBLE)

                    # Additional load for interior beams
                    if 1 < ybay < len(spans_y) + 1:
                        if spans_x[xbay - 1] <= spans_y[ybay - 2]:
                            # Triangular rule
                            load = q * spans_x[xbay - 1] ** 2 / 4 / spans_x[xbay - 1]
                        else:
                            # Trapezoidal rule
                            load = 1 / 4 * q * spans_y[ybay - 2] * (2 * spans_x[xbay - 1] - spans_y[ybay - 2]) / \
                                   spans_x[xbay - 1]

                        # Applying the load
                        op.eleLoad('-ele', beam, '-type', '-beamUniform', -load, self.NEGLIGIBLE)

                else:
                    # Beams along Y direction
                    # Load over a beam
                    control_length = spans_x[xbay - 1] if xbay < len(spans_x) + 1 else spans_x[xbay - 2]
                    if spans_y[ybay - 1] <= control_length:
                        # Triangular rule
                        load = q * spans_y[ybay - 1] ** 2 / 4 / spans_y[ybay - 1]
                    else:
                        # Trapezoidal rule
                        load = 1 / 4 * q * spans_x[xbay - 1] * (2 * spans_y[ybay - 1] - spans_x[xbay - 1]) / \
                               spans_y[ybay - 1]

                    # Applying the load
                    op.eleLoad('-ele', beam, '-type', '-beamUniform', -load, self.NEGLIGIBLE)

                    # Additional load for interior beams
                    if 1 < xbay < len(spans_x) + 1:
                        if spans_y[ybay - 1] <= spans_x[xbay - 2]:
                            # Triangular rule
                            load = q * spans_y[ybay - 1] ** 2 / 4 / spans_y[ybay - 1]
                        else:
                            # Trapezoidal rule
                            load = 1 / 4 * q * spans_x[xbay - 2] * (2 * spans_y[ybay - 1] - spans_x[xbay - 2]) / \
                                   spans_y[ybay - 1]

                        # Applying the load
                        op.eleLoad('-ele', beam, '-type', '-beamUniform', -load, self.NEGLIGIBLE)

    def rigid_diaphragm(self, spans_x, spans_y, nbays_x, nbays_y):
        # Define Rigid floor diaphragm
        # mid-span coordinate for rigid diaphragm
        xa = sum(spans_x) / 2
        ya = sum(spans_y) / 2
        zloc = 0.
        master_nodes = []
        cnt = 0
        for st in range(1, self.i_d.nst + 1):
            # zloc += self.i_d.heights[st-1]
            # masternodetag = int(f"990{st}")
            masternodetag = int(f"{int(nbays_x / 2 + 1)}{int(nbays_y / 2 + 1)}{st}")
            # op.node(masternodetag, xa, ya, zloc)
            # op.fix(masternodetag, 0, 0, 1, 1, 1, 0)
            master_nodes.append(masternodetag)
            # Define rigid diaphragm
            for xbay in range(int(nbays_x + 1)):
                for ybay in range(int(nbays_y + 1)):
                    nodetag = int(f"{1+xbay}{1+ybay}{st}")
                    if nodetag != masternodetag:
                        op.rigidDiaphragm(3, masternodetag, nodetag)
            cnt += 1

    def define_nodes(self):
        """
        defines nodes and fixities
        :return: None
        """
        # Number of bays in x and y directions, spans
        nbays_x = self.i_d.n_bays
        spans_x = self.i_d.spans_x
        nbays_y = len(self.i_d.spans_y)
        spans_y = self.i_d.spans_y

        # Create the nodes
        xloc = 0.
        for xbay in range(int(nbays_x + 1)):
            yloc = 0.
            for ybay in range(int(nbays_y + 1)):
                zloc = 0.
                nodetag = int(f"{1+xbay}{1+ybay}0")
                op.node(nodetag, xloc, yloc, zloc)
                # Fix or pin the base nodes
                if (xloc == 0. or xloc == sum(spans_x)) and (yloc == 0. or yloc == sum(spans_y)):
                    op.fix(nodetag, 1, 1, 1, 1, 1, 1)
                elif 0. < xloc < sum(spans_x) and (yloc == 0. or yloc == sum(spans_y)):
                    op.fix(nodetag, 1, 1, 1, 0, 1, 0)
                elif 0. < yloc < sum(spans_y) and (xloc == 0. or xloc == sum(spans_x)):
                    op.fix(nodetag, 1, 1, 1, 1, 0, 0)
                else:
                    op.fix(nodetag, 1, 1, 1, 0, 0, 0)

                # Generate the remaining nodes
                for st in range(1, self.i_d.nst + 1):
                    zloc += self.i_d.heights[st-1]
                    nodetag = int(f"{1+xbay}{1+ybay}{st}")
                    op.node(nodetag, xloc, yloc, zloc)
                    # Appending nodes for pushover analysis
                    if self.direction == 0:
                        if (ybay == nbays_y or ybay == 0) and xbay == 0:
                            self.spo_nodes.append(nodetag)
                    else:
                        if (xbay == nbays_x or xbay == 0) and ybay == 0:
                            self.spo_nodes.append(nodetag)

                # Increment y coordinate
                if ybay != int(nbays_y):
                    yloc += spans_y[ybay]

            # Increment x coordinate
            if xbay != int(nbays_x):
                xloc += spans_x[xbay]

        if self.pflag:
            print("[NODE] Nodes created!")

    def define_transformations(self):
        """
        defines geometric transformations for beams and columns
        :return: None
        """
        # Geometric transformations
        op.geomTransf("PDelta", self.COL_TRANSF_TAG, 0, 1, 0)
        op.geomTransf("PDelta", self.BEAM_X_TRANSF_TAG, 0, 1, 0)
        op.geomTransf("PDelta", self.BEAM_Y_TRANSF_TAG, -1, 0, 0)

    def lumped_hinge_element(self, et, gt, inode, jnode, my, lp, fc, b, h, ap=1.25, app=0.05, r=0.1, mu_phi=10,
                             pinch_x=0.8, pinch_y=0.5, damage1=0.0, damage2=0.0, beta=0.0, hingeModel=None):
        """
        creates a lumped hinge element
        :param et: int                          Element tag
        :param gt: int                          Geometric transformation tag
        :param inode: int                       i node
        :param jnode: int                       j node
        :param my: float                        Yield moment
        :param lp: float                        Plastic hinge length
        :param fc: float                        Concrete compressive strength, MPa
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
        :param hingeModel: DataFrame            Idealized plastic hinge model parameters
        :return: None
        """
        elastic_modulus = (3320 * np.sqrt(fc) + 6900) * 1000 * self.fstiff
        area = b * h
        iz = b * h ** 3 / 12
        iy = h * b ** 3 / 12
        nu = 0.2
        Gc = elastic_modulus / 2.0 / (1 + nu)
        # Torsional moment of inertia
        if h >= b:
            J = b * h ** 3 * (16 / 3 - 3.36 * h / b * (1 - 1 / 12 * (h / b) ** 4))
        else:
            J = h * b ** 3 * (16 / 3 - 3.36 * b / h * (1 - 1 / 12 * (b / h) ** 4))

        # Material model creation
        hingeMTag1 = int(f"101{et}")
        hingeMTag2 = int(f"102{et}")

        if hingeModel is not None:
            # Moments
            myPos = hingeModel["m1"].iloc[0]
            mpPos = hingeModel["m2"].iloc[0]
            muPos = hingeModel["m3"].iloc[0]
            myNeg = hingeModel["m1Neg"].iloc[0]
            mpNeg = hingeModel["m2Neg"].iloc[0]
            muNeg = hingeModel["m3Neg"].iloc[0]
            # Curvatures
            phiyPos = hingeModel["phi1"].iloc[0]
            phipPos = hingeModel["phi2"].iloc[0]
            phiuPos = hingeModel["phi3"].iloc[0]
            phiyNeg = hingeModel["phi1Neg"].iloc[0]
            phipNeg = hingeModel["phi2Neg"].iloc[0]
            phiuNeg = hingeModel["phi3Neg"].iloc[0]
            # Plastic hinge length
            lp = hingeModel["lp"].iloc[0]

        else:
            eiz = elastic_modulus * iz

            # Curvatures at yield
            phiyNeg = phiyPos = my / eiz

            myNeg = myPos = my
            mpNeg = mpPos = ap * my
            muNeg = muPos = r * my

            phipNeg = phipPos = phiyNeg * mu_phi
            phiuNeg = phiuPos = phipNeg + (mpNeg - muNeg) / (app * my / phiyNeg)

        # Create the uniaxial hysteretic material
        op.uniaxialMaterial("Hysteretic", hingeMTag1, myPos, phiyPos, mpPos, phipPos, muPos, phiuPos,
                            -myNeg, -phiyNeg, -mpNeg, -phipNeg, -muNeg, -phiuNeg,
                            pinch_x, pinch_y, damage1, damage2, beta)
        op.uniaxialMaterial("Hysteretic", hingeMTag2, myPos, phiyPos, mpPos, phipPos, muPos, phiuPos,
                            -myNeg, -phiyNeg, -mpNeg, -phipNeg, -muNeg, -phiuNeg,
                            pinch_x, pinch_y, damage1, damage2, beta)

        # Element creation
        int_tag = int(f"105{et}")
        ph_tag1 = int(f"106{et}")
        ph_tag2 = int(f"107{et}")
        integration_tag = int(f"108{et}")

        # Elastic section
        op.section("Elastic", int_tag, elastic_modulus, area, iy, iz, Gc, J)
        # Create the plastic hinge flexural section about ZZ
        op.section("Uniaxial", ph_tag1, hingeMTag1, 'Mz')
        op.section("Uniaxial", ph_tag2, hingeMTag2, 'Mz')

        if gt == 1:
            # For columns only
            hingeMTag3 = int(f'111{et}')
            hingeMTag4 = int(f'112{et}')
            axialTag = int(f'113{et}')
            aggTag1 = int(f"114{et}")
            aggTag2 = int(f"115{et}")

            # Beam integration
            op.beamIntegration("HingeRadau", integration_tag, aggTag1, lp, aggTag2, lp, int_tag)
            # Create he plastic hinge axial material
            op.uniaxialMaterial("Elastic", axialTag, elastic_modulus * area)

            # Hinge materials
            op.uniaxialMaterial("Hysteretic", hingeMTag3, myPos, phiyPos, mpPos, phipPos, muPos, phiuPos,
                                -myNeg, -phiyNeg, -mpNeg, -phipNeg, -muNeg, -phiuNeg,
                                pinch_x, pinch_y, damage1, damage2, beta)
            op.uniaxialMaterial("Hysteretic", hingeMTag4, myPos, phiyPos, mpPos, phipPos, muPos, phiuPos,
                                -myNeg, -phiyNeg, -mpNeg, -phipNeg, -muNeg, -phiuNeg,
                                pinch_x, pinch_y, damage1, damage2, beta)

            # Aggregate P and Myy behaviour to Mzz behaviour
            op.section("Aggregator", aggTag1, axialTag, "P", hingeMTag3, "My", "-section", ph_tag1)
            op.section("Aggregator", aggTag2, axialTag, "P", hingeMTag4, "My", "-section", ph_tag2)
        else:
            # Beam integration
            op.beamIntegration('HingeRadau', integration_tag, ph_tag1, lp, ph_tag2, lp, int_tag)

        op.element("forceBeamColumn", et, inode, jnode, gt, integration_tag)

    def elastic_analysis_3d(self, analysis, lat_action, grav_loads):
        """

        :param analysis: int                        Analysis type
        :param lat_action: list                     Acting lateral loads in kN
        :param grav_loads: list                     Acting gravity loads in kN/m
        :return: dict                               Demands on structural elements
        """
        # Number of bays in x and y directions, spans
        nbays_x = self.i_d.n_bays
        spans_x = self.i_d.spans_x
        nbays_y = len(self.i_d.spans_y)
        spans_y = self.i_d.spans_y

        # Wipe the model
        self.wipe()
        # Create the model
        op.model('Basic', '-ndm', 3, '-ndf', 6)
        # Create the nodes
        self.define_nodes()

        # Define Rigid floor diaphragm
        self.rigid_diaphragm(spans_x, spans_y, nbays_x, nbays_y)

        # Geometric transformations
        self.define_transformations()

        # Create structural elements
        beams, columns = self.create_elements(nbays_x, nbays_y, elastic=True)

        # Apply lateral loads for static analysis
        # lat_action represents loads for each seismic frame
        if lat_action is not None:
            op.timeSeries("Linear", 1)
            op.pattern("Plain", 1, 1)
            for st in range(1, int(self.i_d.nst + 1)):
                if self.direction == 0:
                    # Along x direction
                    for bay in range(1, int(nbays_x + 2)):
                        op.load(int(f"{bay}1{st}"), lat_action[st - 1] / (nbays_x + 1), self.NEGLIGIBLE,
                                self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)
                        op.load(int(f"{bay}{nbays_y+1}{st}"), lat_action[st - 1] / (nbays_x + 1), self.NEGLIGIBLE,
                                self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)
                else:
                    # Along y direction
                    for bay in range(1, int(nbays_y + 2)):
                        op.load(int(f"1{bay}{st}"), self.NEGLIGIBLE, lat_action[st - 1] / (nbays_y + 1),
                                self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)
                        op.load(int(f"{nbays_x+1}{bay}{st}"), self.NEGLIGIBLE, lat_action[st - 1] / (nbays_y + 1),
                                self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

        # Application of gravity loads for static analysis
        if analysis == 3 or analysis == 5:
            if grav_loads is not None and None not in grav_loads:
                op.timeSeries("Linear", 2)
                op.pattern("Plain", 2, 2)
                # Seismic frames
                for ele in beams["x"]:
                    st = int(str(ele)[-1]) - 1
                    op.eleLoad('-ele', ele, '-type', '-beamUniform', -abs(grav_loads["x"][st]), self.NEGLIGIBLE)
                for ele in beams["y"]:
                    st = int(str(ele)[-1]) - 1
                    op.eleLoad('-ele', ele, '-type', '-beamUniform', -abs(grav_loads["y"][st]), self.NEGLIGIBLE)
                # Gravity frames
                for ele in beams["gravity_x"]:
                    st = int(str(ele)[-1]) - 1
                    op.eleLoad('-ele', ele, '-type', '-beamUniform', -2 * abs(grav_loads["x"][st]), self.NEGLIGIBLE)
                for ele in beams["gravity_y"]:
                    st = int(str(ele)[-1]) - 1
                    op.eleLoad('-ele', ele, '-type', '-beamUniform', -2 * abs(grav_loads["y"][st]), self.NEGLIGIBLE)
            else:
                q_roof = self.i_d.i_d['bldg_ch'][1]
                q_floor = self.i_d.i_d['bldg_ch'][0]
                spans_x = self.i_d.spans_x
                spans_y = self.i_d.spans_y
                op.timeSeries("Linear", 2)
                op.pattern("Plain", 2, 2)

                self.apply_gravity_loads(beams, q_floor, q_roof, spans_x, spans_y)

        # Analysis parameters
        self.static_analysis()

        # Define recorders
        results = self.recorders(nbays_x, nbays_y, direction=self.direction)

        self.wipe()

        return results

    def create_model(self):
        """
        creates the model
        :return: array                          Beam and column element tags
        """
        # Number of bays in x and y directions, spans
        nbays_x = self.i_d.n_bays
        spans_x = self.i_d.spans_x
        nbays_y = len(self.i_d.spans_y)
        spans_y = self.i_d.spans_y

        self.wipe()
        op.model('Basic', '-ndm', 3, '-ndf', 6)

        self.define_nodes()
        self.rigid_diaphragm(spans_x, spans_y, nbays_x, nbays_y)
        self.define_transformations()
        beams, columns = self.create_elements(nbays_x, nbays_y, elastic=False)
        return beams, columns

    def create_elements(self, nbays_x, nbays_y, elastic=True):
        """
        creates elements
        consideration given only for ELFM, so capacities are arbitrary
        :param elastic: bool                        For elastic analysis or inelastic
        :return: list                               Element tags
        """
        # Cross-sections
        cs_x = self.cs["x_seismic"]
        cs_y = self.cs["y_seismic"]
        cs_gr = self.cs["gravity"]

        # Hinge models
        hinge_x = self.hinge["x_seismic"]
        hinge_y = self.hinge["y_seismic"]
        hinge_gr = self.hinge["gravity"]

        # Add column elements
        columns = {"x": [], "y": [], "gravity": []}
        for xbay in range(1, int(nbays_x + 2)):
            for ybay in range(1, int(nbays_y + 2)):
                for st in range(1, int(self.i_d.nst + 1)):
                    # Parameters for elastic static analysis
                    previous_st = st - 1

                    # Columns of seismic frame along x direction
                    if ybay == 1 or ybay == nbays_y + 1:
                        # External columns
                        if xbay == 1 or xbay == nbays_x + 1:
                            b_col = h_col = cs_x[f"he{st}"]
                        else:
                            # Internal columns
                            b_col = h_col = cs_x[f"hi{st}"]

                        # For nonlinear analysis
                        if hinge_x is not None:
                            eleHinge = hinge_x[(hinge_x["Element"] == "Column") & (hinge_x["Bay"] == xbay) & (
                                    hinge_x["Storey"] == st)].reset_index(drop=True)
                        else:
                            eleHinge = None

                    # Columns of seismic frame along y direction
                    elif (xbay == 1 or xbay == nbays_x + 1) and (1 < ybay < nbays_y + 1):
                        # External columns are already created
                        # Internal columns
                        b_col = h_col = cs_y[f"hi{st}"]

                        if hinge_y is not None:
                            eleHinge = hinge_y[(hinge_y["Element"] == "Column") & (hinge_y["Bay"] == ybay) & (
                                    hinge_y["Storey"] == st)].reset_index(drop=True)
                        else:
                            eleHinge = None

                    # Columns of gravity frames
                    else:
                        # Only internal columns
                        b_col = h_col = cs_gr[f"hi{st}"]

                        if hinge_gr is not None:
                            eleHinge = hinge_gr[(hinge_gr["Element"] == "Column") & (hinge_gr["Storey"] ==
                                                                                     st)].reset_index(drop=True)
                        else:
                            eleHinge = None

                    # Column element tag
                    et = int(f"1{xbay}{ybay}{st}")
                    
                    # End nodes of column
                    inode = int(f"{xbay}{ybay}{previous_st}")
                    jnode = int(f"{xbay}{ybay}{st}")

                    # Column cross-section area and moment of inertia
                    area = b_col * h_col
                    inertia = b_col * h_col ** 3 / 12
                    inertiay = h_col * b_col ** 3 / 12
                    if elastic:
                        # Seismic frame along x direction
                        if ybay == 1 or ybay == nbays_y + 1:
                            el_mod = self.get_elastic_modulus(hinge_x, st, "Column", inertia, xbay)
                        elif (xbay == 1 or xbay == nbays_x + 1) and (1 < ybay < nbays_y + 1):
                            # Seismic frame along y direction
                            el_mod = self.get_elastic_modulus(hinge_y, st, "Column", inertia, ybay)
                        else:
                            # Gravity frame columns
                            el_mod = self.get_elastic_modulus(hinge_gr, st, "Column", inertia)

                        nu = 0.2
                        Gc = el_mod / 2.0 / (1 + nu)

                        # Create the elastic element
                        op.element("elasticBeamColumn", et, inode, jnode, area, el_mod, Gc,
                                   self.UBIG, inertiay, inertia, self.COL_TRANSF_TAG)

                    else:
                        # Placeholder
                        my = self.MY_CONSTANT

                        lp = h_col * 1.0  # not important for linear static analysis
                        gt = self.COL_TRANSF_TAG
                        self.lumped_hinge_element(et, gt, inode, jnode, my, lp, self.i_d.fc, b_col, h_col,
                                                  hingeModel=eleHinge)

                    # For recorders
                    if ybay == 1 or ybay == nbays_y + 1:
                        columns["x"].append(et)
                    elif (xbay == 1 or xbay == nbays_x + 1) and (1 < ybay < nbays_y + 1):
                        columns["y"].append(et)
                    else:
                        columns["gravity"].append(et)

                    # Base columns (for recorders, because for some reason base node recorders did not record)
                    if st == 1:
                        self.base_cols.append(et)

        if self.pflag:
            print("[ELEMENT] Columns created!")

        # Add beam elements in X direction
        beams = {"x": [], "y": [], "gravity_x": [], "gravity_y": []}
        for ybay in range(1, int(nbays_y + 2)):
            for xbay in range(1, int(nbays_x + 1)):
                next_bay_x = xbay + 1
                for st in range(1, int(self.i_d.nst + 1)):
                    # Parameters for elastic static analysis
                    if ybay == 1 or ybay == nbays_y + 1:
                        b_beam = cs_x[f"b{st}"]
                        h_beam = cs_x[f"h{st}"]
                    else:
                        b_beam = cs_gr[f"bx{st}"]
                        h_beam = cs_gr[f"hx{st}"]

                    if ybay == 1 or ybay == nbays_y + 1:
                        if hinge_x is not None:
                            eleHinge = hinge_x[(hinge_x["Element"] == "Beam") & (hinge_x["Bay"] == xbay) & (
                                    hinge_x["Storey"] == st)].reset_index(drop=True)
                        else:
                            eleHinge = None
                    else:
                        if hinge_gr is not None:
                            eleHinge = hinge_gr[(hinge_gr["Element"] == "Beam") & (
                                    hinge_gr["Storey"] == st)].reset_index(drop=True)
                        else:
                            eleHinge = None

                    et = int(f"3{xbay}{ybay}{st}")
                    inode = int(f"{xbay}{ybay}{st}")
                    jnode = int(f"{next_bay_x}{ybay}{st}")
                    area = b_beam * h_beam
                    inertia = b_beam * h_beam ** 3 / 12
                    inertiay = h_beam * b_beam ** 3 / 12

                    # Stiffness reduction
                    if elastic:
                        if ybay == 1 or ybay == nbays_y + 1:
                            el_mod = self.get_elastic_modulus(hinge_x, st, "Beam", inertia, xbay)
                        else:
                            # While columns are identical, beams may vary depending on length
                            el_mod = self.get_elastic_modulus(hinge_gr, st, "Beam", inertia, xbay)

                        nu = 0.2
                        Gc = el_mod / 2.0 / (1 + nu)

                        # Create the element
                        op.element("elasticBeamColumn", et, inode, jnode, area, el_mod, Gc,
                                   self.UBIG, inertiay, inertia, self.BEAM_X_TRANSF_TAG)
                    else:
                        lp = 1.0 * h_beam  # not important for linear static analysis
                        my = self.MY_CONSTANT
                        gt = self.BEAM_X_TRANSF_TAG
                        self.lumped_hinge_element(et, gt, inode, jnode, my, lp, self.i_d.fc, b_beam, h_beam,
                                                  hingeModel=eleHinge)

                    # For recorders
                    if ybay == 1 or ybay == nbays_y + 1:
                        beams["x"].append(et)
                    else:
                        beams["gravity_x"].append(et)

        # Add beam elements in Y direction
        for xbay in range(1, int(nbays_x + 2)):
            for ybay in range(1, int(nbays_y + 1)):
                next_bay_y = ybay + 1
                for st in range(1, int(self.i_d.nst + 1)):
                    # Parameters for elastic static analysis
                    if xbay == 1 or xbay == nbays_x + 1:
                        b_beam = cs_y[f"b{st}"]
                        h_beam = cs_y[f"h{st}"]
                    else:
                        b_beam = cs_gr[f"by{st}"]
                        h_beam = cs_gr[f"hy{st}"]

                    if xbay == 1 or xbay == nbays_x + 1:
                        if hinge_y is not None:
                            eleHinge = hinge_y[(hinge_y["Element"] == "Beam") & (hinge_y["Bay"] == ybay) & (
                                    hinge_y["Storey"] == st)].reset_index(drop=True)
                        else:
                            eleHinge = None
                    else:
                        if hinge_gr is not None:
                            eleHinge = hinge_gr[(hinge_gr["Element"] == "Beam") & (hinge_gr["Storey"] ==
                                                                                   st)].reset_index(drop=True)
                        else:
                            eleHinge = None

                    et = int(f"2{xbay}{ybay}{st}")
                    inode = int(f"{xbay}{ybay}{st}")
                    jnode = int(f"{xbay}{next_bay_y}{st}")

                    area = b_beam * h_beam
                    inertia = b_beam * h_beam**3 / 12
                    inertiay = h_beam * b_beam ** 3 / 12

                    # Stiffness reduction
                    if elastic:
                        if xbay == 1 or xbay == nbays_x + 1:
                            el_mod = self.get_elastic_modulus(hinge_y, st, "Beam", inertia, ybay)
                        else:
                            # While columns are identical, beams may vary depending on length
                            el_mod = self.get_elastic_modulus(hinge_gr, st, "Beam", inertia, ybay)

                        nu = 0.2
                        Gc = el_mod / 2.0 / (1 + nu)

                        # Create the element
                        op.element("elasticBeamColumn", et, inode, jnode, area, el_mod, Gc,
                                   self.UBIG, inertiay, inertia, self.BEAM_Y_TRANSF_TAG)
                    else:
                        lp = 1.0 * h_beam  # not important for linear static analysis
                        my = self.MY_CONSTANT
                        gt = self.BEAM_Y_TRANSF_TAG
                        self.lumped_hinge_element(et, gt, inode, jnode, my, lp, self.i_d.fc, b_beam, h_beam,
                                                  hingeModel=eleHinge)

                    # For recorders
                    if xbay == 1 or xbay == nbays_x + 1:
                        beams["y"].append(et)
                    else:
                        beams["gravity_y"].append(et)

        if self.pflag:
            print("[ELEMENT] Beams created!")

        return beams, columns

    def define_masses(self):
        """
        Define masses. Mass should be the total mass of the building, which is then divided by the number of seismic
        frames.
        :return: None
        """
        nbays_x = self.i_d.n_bays
        spans_x = self.i_d.spans_x
        nbays_y = len(self.i_d.spans_y)
        spans_y = self.i_d.spans_y

        # Floor and roof distributed loads in kN/m2
        q_floor = self.i_d.i_d['bldg_ch'][0]
        q_roof = self.i_d.i_d['bldg_ch'][1]

        for st in range(1, self.i_d.nst + 1):
            for xbay in range(1, nbays_x + 2):
                for ybay in range(1, nbays_y + 2):
                    nodetag = int(f"{xbay}{ybay}{st}")

                    # Corner nodes
                    if xbay == 1:
                        if ybay == 1:
                            # Corner node
                            area = spans_x[xbay - 1] * spans_y[ybay - 1] / 4
                        elif ybay == nbays_y + 1:
                            # Corner node
                            area = spans_x[xbay - 1] * spans_y[ybay - 2] / 4
                        else:
                            # Side node
                            area = spans_x[xbay - 1] * (spans_y[ybay - 2] + spans_y[ybay - 1]) / 4

                    elif xbay == nbays_x + 1:
                        if ybay == 1:
                            # Corner node
                            area = spans_x[xbay - 2] * spans_y[ybay - 1] / 4
                        elif ybay == nbays_y + 1:
                            # Corner node
                            area = spans_x[xbay - 2] * spans_y[ybay - 2] / 4
                        else:
                            # Side node
                            area = spans_x[xbay - 2] * (spans_y[ybay - 2] + spans_y[ybay - 1]) / 4

                    else:
                        if ybay == 1:
                            # Side node
                            area = (spans_x[xbay - 2] + spans_x[xbay - 1]) * spans_y[ybay - 1] / 4
                        elif ybay == nbays_y + 1:
                            # Side node
                            area = (spans_x[xbay - 2] + spans_x[xbay - 1]) * spans_y[ybay - 2] / 4
                        else:
                            # Internal node
                            area = (spans_x[xbay - 2] + spans_x[xbay - 1]) * (spans_y[ybay - 2] + spans_y[ybay - 1]) / 4

                    # Mass based on tributary area
                    mass = area * q_roof / 9.81 if st == self.i_d.nst else area * q_floor / 9.81

                    op.mass(nodetag, mass, mass, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

    def ma_analysis(self, num_modes):
        """
        Runs modal analysis
        :param num_modes: DataFrame                 Design solution, cross-section dimensions
        :return: list                               Modal periods
        """
        if self.direction == 0:
            # X direction for recording the modal shapes
            nbays = self.i_d.n_bays
        else:
            # Y direction for recording the modal shapes
            nbays = len(self.i_d.spans_y)

        # Compute the eigenvectors (solver)
        lam = None
        try:
            lam = op.eigen(num_modes)
        except:
            print("[EXCEPTION] Eigensolver failed, trying genBandArpack...")
            try:
                lam = op.eigen('-genBandArpack', num_modes)
            except:
                print("[EXCEPTION] Eigensolver failed, trying fullGenLapack...")
                try:
                    lam = op.eigen('-fullGenLapack', num_modes)
                except:
                    print("[EXCEPTION] Eigensolver failed, trying symmBandLapack...")
                    try:
                        lam = op.eigen('-symmBandLapack', num_modes)
                    except:
                        print("[EXCEPTION] Eigensolver failed.")

        # Record stuff
        op.record()
        # Extract eigenvalues to appropriate arrays
        omega = []
        freq = []
        period = []
        for m in range(num_modes):
            omega.append(np.sqrt(lam[m]))
            freq.append(np.sqrt(lam[m])/2/np.pi)
            period.append(2*np.pi/np.sqrt(lam[m]))

        # Calculate the first modal shape
        modalShape = np.zeros((self.i_d.nst, 2))
        for st in range(self.i_d.nst):
            if self.direction == 0:
                nodetag = int(f"{nbays+1}1{st+1}")
            else:
                nodetag = int(f"1{nbays+1}{st+1}")
            # First mode shape
            modalShape[st, 0] = op.nodeEigenvector(nodetag, 1, 1)
            # Second mode shape
            modalShape[st, 1] = op.nodeEigenvector(nodetag, 2, 2)

        # Normalize the modal shapes (first two modes, most likely assocaited with X and Y directions unless there are
        # large torsional effects)
        modalShape = np.abs(modalShape) / np.max(np.abs(modalShape), axis=0)

        # Calculate the first mode participation factor and effective modal mass
        M = np.zeros((self.i_d.nst, self.i_d.nst))
        for st in range(self.i_d.nst):
            M[st][st] = self.i_d.masses[st] / self.i_d.n_seismic

        # Identity matrix
        identity = np.ones((1, self.i_d.nst))

        gamma = np.zeros(2)
        mstar = np.zeros(2)
        for i in range(2):
            # Modal participation factor
            gamma[i] = (modalShape[:, i].transpose().dot(M)).dot(identity.transpose()) / \
                       (modalShape[:, i].transpose().dot(M)).dot(modalShape[:, i])

            # Modal mass
            mstar[i] = (modalShape[:, i].transpose().dot(M)).dot(identity.transpose())

        return period, modalShape, gamma, mstar

    def singlePush(self, ctrlNode, ctrlDOF, nSteps):
        LoadFactor = [0]
        DispCtrlNode = [0]
        # Test Types
        test = {0: 'EnergyIncr', 1: 'NormDispIncr', 2: 'RelativeEnergyIncr', 3: 'RelativeNormUnbalance',
                4: 'RelativeNormDispIncr', 5: 'NormUnbalance'}
        # Algorithm Types
        algorithm = {0: 'Newton', 1: 'KrylovNewton', 2: 'SecantNewton', 3: 'RaphsonNewton', 4: 'PeriodicNewton',
                     5: 'BFGS',
                     6: 'Broyden', 7: 'NewtonLineSearch'}
        # Integrator Types
        integrator = {0: 'DisplacementControl', 1: 'LoadControl', 2: 'Parallel DisplacementControl',
                      3: 'Minimum Unbalanced Displacement Norm', 4: 'Arc-Length Control'}

        tol = 1e-12  # Set the tolerance to use during the analysis
        iterInit = 10  # Set the initial max number of iterations
        maxIter = 1000  # Set the max number of iterations to use with other integrators

        dU = 0.1 * sum(self.i_d.heights) / nSteps
        op.test(test[0], tol, iterInit)  # lets start with energyincr as test
        op.algorithm(algorithm[0])
        op.integrator(integrator[0], ctrlNode, ctrlDOF, dU)
        op.analysis('Static')

        # Set the initial values to start the while loop
        ok = 0.0
        step = 1.0
        loadf = 1.0

        # This feature of disabling the possibility of having a negative loading has been included.
        # This has been adapted from a similar script by Prof. Garbaggio

        i = 1  # counter for the tests, if the analysis fails starts with new test directly
        j = 0  # counter for the algorithm

        current_test = test[0]
        current_algorithm = algorithm[0]

        while step <= nSteps and ok == 0 and loadf > 0:
            ok = op.analyze(1)

            # If the analysis fails, try the following changes to achieve convergence
            while ok != 0:
                if j == 7:  # this is the final algorithm to try, if the analysis did not converge
                    j = 0
                    i += 1  # reset the algorithm to use
                    if i == 6:  # we have tried everything
                        break

                j += 1  # change the algorithm

                if j < 3:
                    op.algorithm(algorithm[j], '-initial')

                else:
                    op.algorithm(algorithm[j])

                op.test(test[i], tol, maxIter)
                ok = op.analyze(1)
                current_test = test[i]
                current_algorithm = algorithm[j]

            # disp in dir 1
            temp1 = op.nodeDisp(ctrlNode, 1)
            # disp in dir 2
            temp2 = op.nodeDisp(ctrlNode, 2)
            # SRSS of disp in two dirs
            temp = (temp1 ** 2 + temp2 ** 2) ** 0.5
            loadf = op.getTime()
            step += 1.0
            LoadFactor.append(loadf)
            DispCtrlNode.append(temp)

            # Print the current displacement

        if ok != 0:
            print("Displacement Control Analysis is FAILED")
            print('-------------------------------------------------------------------------')

        else:
            print("Displacement Control Analysis is SUCCESSFUL")
            print('-------------------------------------------------------------------------')

        if loadf <= 0:
            print("Stopped because of Load factor below zero:", loadf)
            print('-------------------------------------------------------------------------')

    def spo_algorithm2(self):
        pass

    def spo_algorithm(self, testType, algorithmType, nsteps, iterInit, tol):
        '''Seek for a solution using different test conditions or algorithms'''
        # Set the initial values to start the while loop
        # The feature of disabling the possibility of having a negative loading has been included.
        #   adapted from a similar script by Prof. Garbaggio
        ok = 0
        step = 1
        loadf = 1.0

        # Recording top displacement and base shear
        if self.direction == 0:
            # Along X direction
            topDisp = np.array([op.nodeResponse(self.spo_nodes[-1], 1, 1)])
            baseShear = np.array([0.0])
            for col in self.base_cols:
                baseShear[0] += op.eleForce(col, 1)
        else:
            # Along Y direction
            topDisp = np.array([op.nodeResponse(self.spo_nodes[-1], 2, 1)])
            baseShear = np.array([0.0])
            for col in self.base_cols:
                baseShear[0] += op.eleForce(col, 2)

        while step <= nsteps and ok == 0 and loadf > 0:
            ok = op.analyze(1)
            loadf = op.getTime()
            if ok != 0:
                # print("[STEP] Trying relaxed convergence...")
                op.test(testType, tol * .01, int(iterInit * 50))
                ok = op.analyze(1)
                op.test(testType, tol, iterInit)
            if ok != 0:
                # print("[STEP] Trying Newton with initial then current...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("Newton", "-initialThenCurrent")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                # print("[STEP] Trying ModifiedNewton with initial...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("ModifiedNewton", "-initial")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                # print("[STEP] Trying KrylovNewton...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("KrylovNewton")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                # print("[STEP] Perform a Hail Mary...")
                op.test("FixedNumIter", iterInit)
                ok = op.analyze(1)

            # Recording the displacements and base shear forces
            if self.direction == 0:
                topDisp = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 1, 1)))
            else:
                topDisp = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 2, 1)))

            # topDisp1 = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 1, 1)))
            # topDisp2 = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 2, 1)))
            # topDisp = (topDisp1**2 + topDisp2**2) ** 0.5

            eleForceTemp = 0.
            for col in self.base_cols:
                if self.direction == 0:
                    eleForceTemp += op.eleForce(col, 1)
                else:
                    eleForceTemp += op.eleForce(col, 2)

            baseShear = np.append(baseShear, eleForceTemp)
            loadf = op.getTime()
            step += 1

        # Reverse sign of base_shear (to be positive for better visualization)
        if min(baseShear) < 0.:
            baseShear = -baseShear

        return topDisp, baseShear

    def spo_analysis(self, load_pattern=1, mode_shape=None):
        """
        Starts static pushover analysis
        :param load_pattern: str                    Load pattern shape for static pushover analysis
                                                    0 = Uniform pattern
                                                    1 = Triangular pattern
                                                    2 = First-mode proportional pattern
        :param mode_shape: list                     1st mode shape (compatible with 2nd load pattern)
        :return: None
        """
        # Number of steps
        nsteps = 1000
        tol = 1.e-8
        iterInit = 10
        testType = "EnergyIncr"
        algorithmType = "KrylovNewton"

        '''Load pattern definition'''
        loads = []
        if load_pattern == 0:
            # print("[STEP] Applying Uniform load pattern...")
            for i in self.spo_nodes:
                loads.append(1.)
        elif load_pattern == 1:
            # print("[STEP] Applying triangular load pattern...")
            for h in range(len(self.i_d.heights)):
                if self.i_d.heights[h] != 0.:
                    loads.append(self.i_d.heights[h] / sum(self.i_d.heights[:h+1]))
        elif load_pattern == 2:
            # print("[STEP] Applying 1st mode proportional load pattern...")
            for i in mode_shape:
                loads.append(i)
        else:
            raise ValueError("[EXCEPTION] Wrong load pattern is supplied.")

        # Applying the load pattern
        op.timeSeries("Linear", 4)
        op.pattern("Plain", 400, 4)

        # Pushing the nodes of seismic frames only
        # if self.direction == 0:
        #     for fpush, nodepush in zip(loads, self.spo_nodes[:self.i_d.nst]):
        #         op.load(nodepush, fpush / 2, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
        #                 self.NEGLIGIBLE)
        #
        #     op.pattern("Plain", 401, 4)
        #     for fpush, nodepush in zip(loads, self.spo_nodes[self.i_d.nst:]):
        #         op.load(nodepush, fpush / 2, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
        #                 self.NEGLIGIBLE)
        # else:
        #     for fpush, nodepush in zip(loads, self.spo_nodes[:self.i_d.nst]):
        #         op.load(nodepush, self.NEGLIGIBLE, fpush / 2, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
        #                 self.NEGLIGIBLE)
        #
        #     op.pattern("Plain", 401, 4)
        #     for fpush, nodepush in zip(loads, self.spo_nodes[self.i_d.nst:]):
        #         op.load(nodepush, self.NEGLIGIBLE, fpush / 2, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
        #                 self.NEGLIGIBLE)

        # Pushing all nodes with masses assigned to them
        nbays_x = self.i_d.n_bays
        nbays_y = len(self.i_d.spans_y)
        # Number of nodes per storey
        n_nodes = (nbays_x + 1) * (nbays_y + 1)

        for xbay in range(1, nbays_x + 2):
            for ybay in range(1, nbays_y + 2):
                for st in range(1, self.i_d.nst + 1):
                    nodepush = int(f"{xbay}{ybay}{st}")
                    fpush = loads[st-1]
                    if self.direction == 0:
                        op.load(nodepush, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
                                self.NEGLIGIBLE, self.NEGLIGIBLE)
                    else:
                        op.load(nodepush, self.NEGLIGIBLE, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE,
                                self.NEGLIGIBLE, self.NEGLIGIBLE)

        '''Set initial analysis parameters'''
        op.constraints("Penalty", 1e15, 1e15)
        # op.constraints("Transformation")
        op.numberer("RCM")
        op.system("UmfPack")
        op.test(testType, tol, iterInit, 0)
        op.algorithm(algorithmType)
        op.integrator("DisplacementControl", self.spo_nodes[-1], self.direction + 1,
                      0.1 * sum(self.i_d.heights) / nsteps)
        op.analysis("Static")

        topDisp, baseShear = self.spo_algorithm(testType, algorithmType, nsteps, iterInit, tol)

        return topDisp, baseShear


if __name__ == "__main__":

    from pathlib import Path
    from client.input import Input
    import pandas as pd
    import pickle

    directory = Path.cwd().parents[1] / ".applications/LOSS Validation Manuscript/case21"

    actionx = directory.parents[0] / "sample" / "actionx.csv"
    actiony = directory.parents[0] / "sample" / "actiony.csv"
    csx = directory / "Cache/solution_cache_x.csv"
    csy = directory / "Cache/solution_cache_y.csv"
    csg = directory / "gravity_cs.csv"
    input_file = directory.parents[0] / "case21/ipbsd_input.csv"
    hinge_models = Path.cwd().parents[0] / "tempHinge.pickle"
    direction = 0
    modalShape = [0.37, 0.64, 0.87, 1.]

    # Read the cross-section files
    idx_x = 41
    idx_y = 20

    csx = pd.read_csv(csx, index_col=0).iloc[idx_x]
    csy = pd.read_csv(csy, index_col=0).iloc[idx_y]
    csg = pd.read_csv(csg, index_col=0).iloc[0]

    cs = {"x_seismic": csx, "y_seismic": csy, "gravity": csg}

    actionx = pd.read_csv(actionx)
    actiony = pd.read_csv(actiony)

    lat_action = list(actiony["Fi"])

    # Hinge models
    with open(hinge_models, 'rb') as file:
        hinge = pickle.load(file)

    hinge_elastic = {"x_seismic": None, "y_seismic": None, "gravity": None}
    fstiff = 0.5

    # Read input data
    data = Input()
    data.read_inputs(input_file)

    # analysis = OpenSeesRun3D(data, cs, hinge=hinge_elastic, direction=direction, fstiff=fstiff)
    # results = analysis.elastic_analysis_3d(analysis=3, lat_action=lat_action, grav_loads=None)

    # ma = OpenSeesRun3D(data, cs, hinge=hinge, direction=direction, fstiff=fstiff)
    # ma.create_model()
    # ma.define_masses()
    # model_periods, modalShape, gamma, mstar = ma.ma_analysis(3)
    # ma.wipe()

    spo = OpenSeesRun3D(data, cs, fstiff, hinge=hinge, direction=direction)
    spo.create_model()
    # spo.define_masses()
    topDisp, baseShear = spo.spo_analysis(load_pattern=2, mode_shape=modalShape)
    spo.wipe()

    # spo_results = {"d": topDisp, "v": baseShear}
    # with open(f"temp_spo.pickle", 'wb') as handle:
    #     pickle.dump(spo_results, handle)
