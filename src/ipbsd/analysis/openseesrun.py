"""
Runs OpenSees software for structural analysis (in future may be changed with anastruct or other software)
For now only ELFM, pushover to be added later if necessary.
The idea is to keep it as simple as possible, and with fewer runs of this class.
This is  linear elastic analysis in order to obtain demands on structural components for further detailing
"""
import openseespy.opensees as op
import numpy as np


class OpenSeesRun:
    def __init__(self, data, cross_sections, fstiff=0.5, hinge=None, pflag=False, direction=0, system="perimeter",
                 flag3d=True):
        """
        initializes model creation for analysing
        :param data: dict                     Provided input arguments for the framework
        :param cross_sections: dict             DataFrames of Cross-sections of the solution
        :param fstiff: float                    Stiffness reduction factor
        :param hinge: dict                      DataFrames of Idealized plastic hinge model parameters
        :param pflag: bool                      Print info
        :param direction: bool                  0 for x direction, 1 for y direction
        :param system: str                      System type (perimeter or space)
        :param flag3d: bool                     True=3D model, False=2D model
        """
        self.NEGLIGIBLE = 1.e-9
        self.UBIG = 1.e10

        self.data = data
        self.cross_sections = cross_sections
        self.fstiff = fstiff
        self.pflag = pflag
        self.hinge = hinge
        # direction must be 0 for 2D modelling
        self.direction = direction
        self.system = system
        self.flag3d = flag3d
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
        self.MY_CONSTANT = 4000.

    @staticmethod
    def wipe():
        """
        wipes model
        :return: None
        """
        op.wipe()

    def run_static_analysis(self):
        if self.flag3d:
            dgravity = 1.0 / 1
            op.integrator("LoadControl", dgravity)
            op.numberer("RCM")
            op.system("UmfPack")
            op.constraints("Penalty", 1.0e15, 1.0e15)
            op.test("EnergyIncr", 1.0e-8, 10)
        else:
            op.integrator("LoadControl", 0.1)
            op.numberer("Plain")
            op.system("BandGeneral")
            op.constraints("Plain")
            op.test("NormDispIncr", 1.0e-8, 6)
        op.algorithm("Newton")
        op.analysis("Static")
        op.analyze(1)
        op.loadConst("-time", 0.0)

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
            el_mod = (3320 * np.sqrt(self.data.fc) + 6900) * 1000 * self.fstiff

        return el_mod

    @staticmethod
    def record_results(results, frame, element, st, bay, et, moment, axial, shear, ybay=None):

        if ybay:
            if element == "Columns":
                results[frame][element]["M"][st][bay][ybay] = max(abs(op.eleForce(et, moment[1])),
                                                                  abs(op.eleForce(et, moment[0])))
            else:
                results[frame][element]["M"]["Pos"][st][bay][ybay] = abs(op.eleForce(et, moment[1]))
                results[frame][element]["M"]["Neg"][st][bay][ybay] = abs(op.eleForce(et, moment[0]))
            results[frame][element]["N"][st][bay][ybay] = max(op.eleForce(et, axial[0]),
                                                              op.eleForce(et, axial[1]), key=abs)
            results[frame][element]["V"][st][bay][ybay] = max(abs(op.eleForce(et, shear[0])),
                                                              abs(op.eleForce(et, shear[1])))

        else:
            if element == "Columns":
                results[frame][element]["M"][st][bay] = max(abs(op.eleForce(et, moment[1])),
                                                            abs(op.eleForce(et, moment[0])))
            else:
                results[frame][element]["M"]["Pos"][st][bay] = abs(op.eleForce(et, moment[1]))
                results[frame][element]["M"]["Neg"][st][bay] = abs(op.eleForce(et, moment[0]))
            results[frame][element]["N"][st][bay] = max(op.eleForce(et, axial[0]),
                                                        op.eleForce(et, axial[1]), key=abs)
            results[frame][element]["V"][st][bay] = max(abs(op.eleForce(et, shear[0])),
                                                        abs(op.eleForce(et, shear[1])))

        return results

    def record(self, nbays_x, nbays_y, direction=0):

        # Indices for recorders
        # direction must be 0 for 2D modelling
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
        # Only bx_seismic and cx_seismic are of interest for 2D modelling
        bx_seismic = np.zeros((self.data.nst, nbays_x))
        by_seismic = np.zeros((self.data.nst, nbays_y))
        cx_seismic = np.zeros((self.data.nst, nbays_x + 1))
        cy_seismic = np.zeros((self.data.nst, nbays_y + 1))

        # For gravity frame elements only the max demands will be used for uniform design
        bx_gravity = np.zeros((self.data.nst, nbays_x, nbays_y - 1))
        by_gravity = np.zeros((self.data.nst, nbays_x - 1, nbays_y))
        c_gravity = np.zeros((self.data.nst, nbays_x - 1, nbays_y - 1))

        # Global dictionary variable to store all demands
        # Only x_seismic is relevant for 2D modelling
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
            for st in range(self.data.nst):
                if self.flag3d:
                    et = int(f"3{bay+1}1{st+1}")
                else:
                    et = int(f"1{bay}{st}")

                results = self.record_results(results, "x_seismic", "Beams", st, bay, et, midx, nidx_b, vidx_b)

        # Columns
        # Columns of seismic frame along x direction
        # Columns [Vx, Vy, N, Mx, My, Mz]; jNode [Vx, Vy, N, Mx, My, Mz]
        # Columns for X direction demand estimations [V, 2, N, 3, M, 6]
        # Columns for Y direction demand estimations [0, V, N, M, 5, 6]
        for bay in range(nbays_x + 1):
            for st in range(self.data.nst):
                if self.flag3d:
                    et = int(f"1{bay + 1}1{st + 1}")
                else:
                    et = int(f"2{bay}{st}")

                results = self.record_results(results, "x_seismic", "Columns", st, bay, et, midx, nidx_c, vidx_c)

        if not self.flag3d:
            # return if 2D modelling was selected, otherwise continue recording
            return results["x_seismic"]

        # Remaining Beams
        # Beams of seismic frame along Y direction
        for bay in range(nbays_y):
            for st in range(self.data.nst):
                et = int(f"21{bay+1}{st+1}")
                results = self.record_results(results, "y_seismic", "Beams", st, bay, et, midx, nidx_b, vidx_b)

        # Beams of gravity frames along X direction
        for ybay in range(nbays_y - 1):
            for xbay in range(nbays_x):
                for st in range(self.data.nst):
                    et = int(f"3{xbay+1}{ybay+2}{st+1}")
                    results = self.record_results(results, "gravity", "Beams_x", st, xbay, et, midx, nidx_b, vidx_b,
                                                  ybay)

        # Beams of gravity frames along Y direction
        for xbay in range(nbays_x - 1):
            for ybay in range(nbays_y):
                for st in range(self.data.nst):
                    et = int(f"2{xbay + 2}{ybay + 1}{st + 1}")
                    results = self.record_results(results, "gravity", "Beams_y", st, xbay, et, midx, nidx_b, vidx_b,
                                                  ybay)

        # Remaining Columns
        # Columns of seismic frame along y direction
        for bay in range(nbays_y + 1):
            for st in range(self.data.nst):
                et = int(f"11{bay+1}{st+1}")
                results = self.record_results(results, "y_seismic", "Columns", st, bay, et, midx, nidx_c, vidx_c)

        # Columns of gravity frames
        for xbay in range(nbays_x - 1):
            for ybay in range(nbays_y - 1):
                for st in range(self.data.nst):
                    et = int(f"1{xbay+2}{ybay+2}{st+1}")
                    results = self.record_results(results, "gravity", "Columns", st, xbay, et, midx, nidx_c, vidx_c,
                                                  ybay)

        return results

    def get_load(self, beam, spans, control_length, bay, q, distributed, direction=0):
        if spans[bay - 1] <= control_length:
            # Triangular rule
            load = q * spans[bay - 1] ** 2 / 4 / spans[bay - 1]
        else:
            # Trapezoidal rule
            load = 1 / 4 * q * control_length * (2 * spans[bay - 1] - control_length) / \
                   spans[bay - 1]
        load = round(load, 2)

        # End nodes connecting the beam
        if direction == 0:
            inode = beam - 3000
            jnode = beam - 3000 + 100
        else:
            inode = beam - 2000
            jnode = beam - 2000 + 10

        # apply the load
        if distributed:
            # If the load is applied uniformly
            op.eleLoad('-ele', beam, '-type', '-beamUniform', load, self.NEGLIGIBLE)
        else:
            # If the load is applied as point load at the end nodes of the beam
            op.load(inode, self.NEGLIGIBLE, self.NEGLIGIBLE, -load * spans[bay - 1] / 2,
                    self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)
            op.load(jnode, self.NEGLIGIBLE, self.NEGLIGIBLE, -load * spans[bay - 1] / 2,
                    self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

    def apply_lateral(self, lateral_action, nbay, nodes, bay, st, direction=0):
        if direction == 0:
            x = lateral_action[st - 1] / nodes
            y = self.NEGLIGIBLE

            # element IDs
            ele1 = int(f"{bay}1{st}")
            ele2 = int(f"{bay}{nbay + 1}{st}")
        else:
            x = self.NEGLIGIBLE
            y = lateral_action[st - 1] / nodes

            # element IDs
            ele1 = int(f"1{bay}{st}")
            ele2 = int(f"{nbay+1}{bay}{st}")

        if not self.flag3d:
            # 2D model
            op.load(int(f"{bay}{st}"), y, self.NEGLIGIBLE, self.NEGLIGIBLE)
            return

        op.load(ele1, x, y, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)
        op.load(ele2, x, y, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

        if self.system == "space":
            for ybay in range(2, int(nbay + 1)):
                if direction == 0:
                    ele = int(f"{bay}{ybay}{st}")
                else:
                    ele = int(f"{ybay}{bay}{st}")

                op.load(ele, x, y, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

    def get_quantities(self):
        # Number of bays in x and y directions, spans
        nbays_x = self.data.n_bays
        spans_x = self.data.spans_x
        nbays_y = len(self.data.spans_y)
        spans_y = self.data.spans_y

        return nbays_x, spans_x, nbays_y, spans_y

    def apply_gravity_loads(self, beams, distributed=True, action=None):
        """
        Action required for 2D modelling
        :param beams: dict
        :param distributed: bool
        :param action: DataFrame
        :return:
        """
        op.timeSeries("Linear", 2)
        op.pattern("Plain", 2, 2)

        if action is not None:
            # starts applying loads for 2D modelling
            for ele in beams:
                storey = int(str(ele)[-1]) - 1
                op.eleLoad('-ele', ele, '-type', '-beamUniform', action[storey], self.NEGLIGIBLE)

            # break the function
            return

        # If 2D modelling did not initiate, continue for 3D modelling
        # get distributed loads and span dimensions
        q_roof = self.data.inputs['loads'][1]
        q_floor = self.data.inputs['loads'][0]
        spans_x = self.data.spans_x
        spans_y = self.data.spans_y

        for d in beams:
            for beam in beams[d]:
                st = int(str(beam)[-1])
                xbay = int(str(beam)[1])
                ybay = int(str(beam)[2])

                # Distributed loads (kN/m2)
                if st != self.data.nst:
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
                    self.get_load(beam, spans_x, control_length, xbay, q, distributed)

                    # Additional load for interior beams
                    if 1 < ybay < len(spans_y) + 1:
                        control_length = spans_y[ybay - 2]
                        self.get_load(beam, spans_x, control_length, xbay, q, distributed)

                else:
                    # Beams along Y direction
                    # Load over a beam
                    control_length = spans_x[xbay - 1] if xbay < len(spans_x) + 1 else spans_x[xbay - 2]
                    self.get_load(beam, spans_y, control_length, ybay, q, distributed, 1)

                    # Additional load for interior beams
                    if 1 < xbay < len(spans_x) + 1:
                        control_length = spans_x[xbay - 2]
                        self.get_load(beam, spans_y, control_length, ybay, q, distributed, 1)

    def create_rigid_diaphragm(self, spans_x, spans_y, nbays_x, nbays_y):
        # Define Rigid floor diaphragm
        # mid-span coordinate for rigid diaphragm
        xa = sum(spans_x) / 2
        ya = sum(spans_y) / 2
        zloc = 0.
        master_nodes = []
        cnt = 0
        for st in range(1, self.data.nst + 1):
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
        Defines nodes and fixities
        :return: None
        """
        # Number of bays in x and y directions, spans
        nbays_x, spans_x, nbays_y, spans_y = self.get_quantities()

        # Create the nodes in a 3D or 2D space
        # x coordinate
        xloc = 0.
        for xbay in range(int(nbays_x + 1)):
            # y coordinate
            yloc = 0.

            if self.flag3d:
                for ybay in range(int(nbays_y + 1)):
                    # z coordinate
                    zloc = 0.
                    nodetag = int(f"{1+xbay}{1+ybay}0")
                    op.node(nodetag, xloc, yloc, zloc)
                    if self.system == "perimeter":
                        # Fix or pin the base nodes depending on location
                        if (xloc == 0. or xloc == sum(spans_x)) and (yloc == 0. or yloc == sum(spans_y)):
                            op.fix(nodetag, 1, 1, 1, 1, 1, 1)
                        elif 0. < xloc < sum(spans_x) and (yloc == 0. or yloc == sum(spans_y)):
                            op.fix(nodetag, 1, 1, 1, 0, 1, 0)
                        elif 0. < yloc < sum(spans_y) and (xloc == 0. or xloc == sum(spans_x)):
                            op.fix(nodetag, 1, 1, 1, 1, 0, 0)
                        else:
                            op.fix(nodetag, 1, 1, 1, 0, 0, 0)
                    else:
                        # Fix all base nodes
                        op.fix(nodetag, 1, 1, 1, 1, 1, 1)

                    # Generate the remaining nodes
                    for st in range(1, self.data.nst + 1):
                        zloc += self.data.heights[st - 1]
                        nodetag = int(f"{1+xbay}{1+ybay}{st}")
                        op.node(nodetag, xloc, yloc, zloc)

                        # Appending nodes for pushover analysis
                        if self.direction == 0:
                            if (ybay == nbays_y or ybay == 0) and xbay == nbays_x:
                                self.spo_nodes.append(nodetag)
                        else:
                            if (xbay == nbays_x or xbay == 0) and ybay == nbays_y:
                                self.spo_nodes.append(nodetag)

                    # Increment y coordinate
                    if ybay != int(nbays_y):
                        yloc += spans_y[ybay]

            else:
                nodetag = int(str(1 + xbay) + '0')
                self.base_nodes.append(nodetag)
                op.node(nodetag, xloc, 0, yloc)
                op.fix(nodetag, 1, 1, 1, 1, 1, 1)
                for st in range(1, self.data.nst + 1):
                    yloc += self.data.heights[st - 1]
                    nodetag = int(str(1 + xbay) + str(st))
                    op.node(nodetag, xloc, 0, yloc)
                    if xbay == self.data.n_bays:
                        self.spo_nodes.append(nodetag)

            # Increment x coordinate
            if xbay != int(nbays_x):
                xloc += spans_x[xbay]

        # fix out of plane for 2D models
        if not self.flag3d:
            op.fixY(0.0, 0, 1, 0, 0, 0, 1)

        if self.pflag:
            print("[NODE] Nodes created!")

    def define_transformations(self):
        """
        Defines geometric transformations for beams and columns
        :return: None
        """
        # Geometric transformations
        op.geomTransf("PDelta", self.COL_TRANSF_TAG, 0, 1, 0)
        op.geomTransf("PDelta", self.BEAM_X_TRANSF_TAG, 0, 1, 0)
        if self.flag3d:
            op.geomTransf("PDelta", self.BEAM_Y_TRANSF_TAG, -1, 0, 0)

    def generate_lumped_hinge_element(self, et, gt, inode, jnode, my, lp, fc, b, h, ap=1.25, app=0.05, r=0.1, mu_phi=10,
                                      pinch_x=0.8, pinch_y=0.5, damage1=0.0, damage2=0.0, beta=0.0, hinge_model=None):
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
        :param hinge_model: DataFrame           Idealized plastic hinge model parameters
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

        if hinge_model is not None:
            # Moments
            myPos = hinge_model["m1"].iloc[0]
            mpPos = hinge_model["m2"].iloc[0]
            muPos = hinge_model["m3"].iloc[0]
            myNeg = hinge_model["m1Neg"].iloc[0]
            mpNeg = hinge_model["m2Neg"].iloc[0]
            muNeg = hinge_model["m3Neg"].iloc[0]
            # Curvatures
            phiyPos = hinge_model["phi1"].iloc[0]
            phipPos = hinge_model["phi2"].iloc[0]
            phiuPos = hinge_model["phi3"].iloc[0]
            phiyNeg = hinge_model["phi1Neg"].iloc[0]
            phipNeg = hinge_model["phi2Neg"].iloc[0]
            phiuNeg = hinge_model["phi3Neg"].iloc[0]
            # Plastic hinge length
            lp = hinge_model["lp"].iloc[0]

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

        if gt == 1 and self.flag3d:
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

    def run_elastic_analysis(self, analysis, lateral_action=None, grav_loads=None):
        """

        :param analysis: int                        Analysis type
        :param lateral_action: list                 Acting lateral loads in kN
        :param grav_loads: list                     Acting gravity loads in kN/m
        :return: dict                               Demands on structural elements
        """
        # Number of bays in x and y directions, spans
        nbays_x, spans_x, nbays_y, spans_y = self.get_quantities()

        # create the model
        beams, columns = self.create_model(elastic=True)

        # Apply lateral loads for static analysis
        # lateral_action represents loads for each seismic frame
        if lateral_action is not None:
            if self.system == "perimeter" or not self.flag3d:
                n_nodes_x = nbays_x + 1
                n_nodes_y = nbays_y + 1
            else:
                n_nodes_x = n_nodes_y = (nbays_y + 1) * (nbays_x + 1)

            op.timeSeries("Linear", 1)
            op.pattern("Plain", 1, 1)
            for st in range(1, int(self.data.nst + 1)):
                if self.direction == 0:
                    # Along x direction
                    for bay in range(1, int(nbays_x + 2)):
                        self.apply_lateral(lateral_action, nbays_y, n_nodes_x, bay, st)
                else:
                    # Along y direction
                    for bay in range(1, int(nbays_y + 2)):
                        self.apply_lateral(lateral_action, nbays_x, n_nodes_y, bay, st, direction=1)

        # Application of gravity loads for static analysis
        if analysis == 3 or analysis == 5:
            if grav_loads is not None and None not in grav_loads:
                op.timeSeries("Linear", 2)
                op.pattern("Plain", 2, 2)
                if self.flag3d:
                    # Seismic frames
                    for ele in beams["x"]:
                        st = int(str(ele)[-1]) - 1
                        op.eleLoad('-ele', ele, '-type', '-beamUniform', abs(grav_loads["x"][st]), self.NEGLIGIBLE)
                    for ele in beams["y"]:
                        st = int(str(ele)[-1]) - 1
                        op.eleLoad('-ele', ele, '-type', '-beamUniform', abs(grav_loads["y"][st]), self.NEGLIGIBLE)
                    # Gravity frames
                    for ele in beams["gravity_x"]:
                        st = int(str(ele)[-1]) - 1
                        op.eleLoad('-ele', ele, '-type', '-beamUniform', 2 * abs(grav_loads["x"][st]), self.NEGLIGIBLE)
                    for ele in beams["gravity_y"]:
                        st = int(str(ele)[-1]) - 1
                        op.eleLoad('-ele', ele, '-type', '-beamUniform', 2 * abs(grav_loads["y"][st]), self.NEGLIGIBLE)
                else:
                    for ele in beams:
                        storey = int(str(ele)[-1]) - 1
                        op.eleLoad('-ele', ele, '-type', '-beamUniform', -abs(grav_loads[storey]))
            else:
                self.apply_gravity_loads(beams)

        # Analysis parameters
        self.run_static_analysis()

        # Define recorders
        results = self.record(nbays_x, nbays_y, direction=self.direction)

        self.wipe()

        return results

    def create_model(self, gravity=False, elastic=False):
        """
        creates the model
        :param gravity: bool                    Apply gravity loads
        :param elastic: bool                    Runs elastic analysis
        :return: array                          Beam and column element tags
        """
        # Number of bays in x and y directions, spans
        nbays_x, spans_x, nbays_y, spans_y = self.get_quantities()

        self.wipe()
        if self.flag3d:
            op.model('Basic', '-ndm', 3, '-ndf', 6)
            loads = None
        else:
            op.model('Basic', '-ndm', 2, '-ndf', 3)
            loads = [self.data.w_seismic["floor"]] * (self.data.nst - 1) + [self.data.w_seismic["roof"]]

        self.define_transformations()
        self.define_nodes()
        beams, columns = self.create_elements(nbays_x, nbays_y)

        if not elastic:
            if self.flag3d:
                self.create_rigid_diaphragm(spans_x, spans_y, nbays_x, nbays_y)

            if gravity:
                self.apply_gravity_loads(beams, action=loads)
                # Static analysis
                self.run_static_analysis()

        return beams, columns

    @staticmethod
    def get_hinge_model_column(ybay, nbays_y, xbay, nbays_x, cs_x, hinge_x, cs_y, hinge_y, cs_gr, hinge_gr, st):

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

        return eleHinge, b_col, h_col

    @staticmethod
    def get_hinge_model_beam(direction, bay, nbays, ybay, cs_x, hinge_x, cs_gr, hinge_gr, st):

        if bay == 1 or bay == nbays + 1:
            b_beam = cs_x[f"b{st}"]
            h_beam = cs_x[f"h{st}"]
        else:
            b_beam = cs_gr[f"b{direction}{st}"]
            h_beam = cs_gr[f"h{direction}{st}"]

        if bay == 1 or bay == nbays + 1:
            if hinge_x is not None:
                eleHinge = hinge_x[(hinge_x["Element"] == "Beam") & (hinge_x["Bay"] == ybay) & (
                        hinge_x["Storey"] == st)].reset_index(drop=True)
            else:
                eleHinge = None
        else:
            if hinge_gr is not None:
                if direction == "x":
                    tag = 0
                else:
                    tag = 1

                eleHinge = hinge_gr[(hinge_gr["Element"] == "Beam") & (hinge_gr["Storey"] == st)
                                    & (hinge_gr["Direction"] == tag)].reset_index(drop=True)
            else:
                eleHinge = None

        return eleHinge, b_beam, h_beam

    def create_elements(self, nbays_x, nbays_y):
        """
        creates elements
        consideration given only for ELFM, so capacities are arbitrary
        :return: list                               Element tags
        """
        if self.flag3d:
            # Cross-sections
            cs_x = self.cross_sections["x_seismic"]
            cs_y = self.cross_sections["y_seismic"]
            cs_gr = self.cross_sections["gravity"]

            # Hinge models
            hinge_x = self.hinge["x_seismic"]
            hinge_y = self.hinge["y_seismic"]
            hinge_gr = self.hinge["gravity"]

            # For storing
            columns = {"x": [], "y": [], "gravity": []}
            beams = {"x": [], "y": [], "gravity_x": [], "gravity_y": []}
        else:
            cs_x, cs_y, cs_gr, hinge_y, hinge_gr = None, None, None, None, None
            hinge_x = self.hinge
            cs_x = self.cross_sections
            columns = []
            beams = []

        # Placeholders
        my = self.MY_CONSTANT
        lp = 0.6

        # Add column elements
        for xbay in range(1, int(nbays_x + 2)):
            for st in range(1, int(self.data.nst + 1)):
                # previous storey level
                previous_st = st - 1

                if not self.flag3d:
                    # 2D model
                    eleHinge, b_col, h_col = self.get_hinge_model_column(1, 1, xbay, nbays_x, cs_x, hinge_x, cs_y,
                                                                         hinge_y, cs_gr, hinge_gr, st)

                    et = int(f"2{xbay}{st}")
                    inode = int(f"{xbay}{previous_st}")
                    jnode = int(f"{xbay}{st}")
                    self.generate_lumped_hinge_element(et, self.COL_TRANSF_TAG, inode, jnode, my, lp, self.data.fc, b_col,
                                                       h_col, hinge_model=eleHinge)

                    # append column ID
                    columns.append(et)
                    # Base columns (for recorders, because for some reason base node recorders did not record)
                    if st == 1:
                        self.base_cols.append(et)

                else:
                    # 3D model
                    for ybay in range(1, int(nbays_y + 2)):
                        # Parameters for elastic static analysis
                        # get hinge model and cross-section dimensions
                        eleHinge, b_col, h_col = self.get_hinge_model_column(ybay, nbays_y, xbay, nbays_x, cs_x,
                                                                             hinge_x, cs_y, hinge_y, cs_gr, hinge_gr, st)

                        # Column element tag
                        et = int(f"1{xbay}{ybay}{st}")

                        # End nodes of column
                        inode = int(f"{xbay}{ybay}{previous_st}")
                        jnode = int(f"{xbay}{ybay}{st}")

                        self.generate_lumped_hinge_element(et, self.COL_TRANSF_TAG, inode, jnode, my, lp, self.data.fc, b_col,
                                                           h_col, hinge_model=eleHinge)

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
        for ybay in range(1, int(nbays_y + 2)):
            for st in range(1, int(self.data.nst + 1)):
                if not self.flag3d:
                    # 2D model
                    # Parameters for elastic static analysis
                    eleHinge, b_beam, h_beam = self.get_hinge_model_beam("x", 1, 1, ybay, cs_x, hinge_x, cs_gr,
                                                                         hinge_gr, st)
                    # element and node tags
                    et = int(f"1{ybay}{st}")
                    inode = int(f"{ybay}{st}")
                    next_bay = ybay + 1
                    jnode = int(f"{next_bay}{st}")
                    self.generate_lumped_hinge_element(et, self.BEAM_X_TRANSF_TAG, inode, jnode, my, lp, self.data.fc, b_beam,
                                                       h_beam, hinge_model=eleHinge)
                    # Add beam element ID
                    beams.append(et)

                else:
                    # 3D model
                    for xbay in range(1, int(nbays_x + 1)):
                        next_bay_x = xbay + 1

                        # Parameters for elastic static analysis
                        eleHinge, b_beam, h_beam = self.get_hinge_model_beam("x", ybay, nbays_y, xbay, cs_x, hinge_x,
                                                                             cs_gr, hinge_gr, st)
                        # element and node tags
                        et = int(f"3{xbay}{ybay}{st}")
                        inode = int(f"{xbay}{ybay}{st}")
                        jnode = int(f"{next_bay_x}{ybay}{st}")

                        # Placeholders
                        self.generate_lumped_hinge_element(et, self.BEAM_X_TRANSF_TAG, inode, jnode, my, lp, self.data.fc,
                                                           b_beam, h_beam, hinge_model=eleHinge)

                        # For recorders
                        if ybay == 1 or ybay == nbays_y + 1:
                            beams["x"].append(et)
                        else:
                            beams["gravity_x"].append(et)

        # Add beam elements in Y direction
        for xbay in range(1, int(nbays_x + 2)):
            for ybay in range(1, int(nbays_y + 1)):
                next_bay_y = ybay + 1
                for st in range(1, int(self.data.nst + 1)):
                    # Parameters for elastic static analysis
                    eleHinge, b_beam, h_beam = self.get_hinge_model_beam("x", xbay, nbays_x, ybay, cs_y, hinge_y, cs_gr,
                                                                         hinge_gr, st)
                    # element and node tags
                    et = int(f"2{xbay}{ybay}{st}")
                    inode = int(f"{xbay}{ybay}{st}")
                    jnode = int(f"{xbay}{next_bay_y}{st}")

                    # Placeholders
                    self.generate_lumped_hinge_element(et, self.BEAM_Y_TRANSF_TAG, inode, jnode, my, lp, self.data.fc, b_beam,
                                                       h_beam, hinge_model=eleHinge)

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
        nbays_x, spans_x, nbays_y, spans_y = self.get_quantities()

        # Floor and roof distributed loads in kN/m2
        q_floor = self.data.inputs['loads'][0]
        q_roof = self.data.inputs['loads'][1]

        for st in range(1, self.data.nst + 1):
            for xbay in range(1, nbays_x + 2):
                if self.flag3d:
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
                        mass = area * q_roof / 9.81 if st == self.data.nst else area * q_floor / 9.81

                        op.mass(nodetag, mass, mass, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE)

                else:
                    if xbay == 1 or xbay == nbays_x + 1:
                        # corner nodes
                        mass = self.data.masses[st - 1] / (2 * nbays_x) / self.data.n_seismic
                    else:
                        mass = self.data.masses[st - 1] / nbays_x / self.data.n_seismic
                    op.mass(int(f"{xbay}{st}"), mass, self.NEGLIGIBLE, mass, self.NEGLIGIBLE, self.NEGLIGIBLE,
                            self.NEGLIGIBLE)

    def run_modal_analysis(self, num_modes):
        """
        Runs modal analysis
        :param num_modes: DataFrame                 Design solution, cross-section dimensions
        :return: list                               Modal periods
        """
        if self.direction == 0:
            # X direction for recording the modal shapes
            nbays = self.data.n_bays
        else:
            # Y direction for recording the modal shapes
            nbays = len(self.data.spans_y)

        # Get all node tags
        nodes = op.getNodeTags()

        # Check problem size (2D or 3D)
        ndm = len(op.nodeCoord(nodes[0]))

        # Initialize computation of total masses
        if ndm == 3:
            # 3D building
            ndf_max = 6
            total_mass = np.array([0] * 6)
        else:
            # 2D frame
            ndf_max = 3
            total_mass = np.array([0] * 3)

        # Get the total masses
        for node in nodes:
            indf = len(op.nodeDisp(node))
            for i in range(indf):
                total_mass[i] += op.nodeMass(node, i + 1)

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

        # Results for each mode
        mode_data = np.zeros((num_modes, 4))
        mode_MPM = np.zeros((num_modes, ndf_max))
        mode_L = np.zeros((num_modes, ndf_max))

        # Extract eigenvalues to appropriate arrays
        omega = []
        freq = []
        period = []
        for m in range(num_modes):
            omega.append(np.sqrt(lam[m]))
            freq.append(np.sqrt(lam[m]) / 2 / np.pi)
            period.append(2 * np.pi / np.sqrt(lam[m]))
            mode_data[m, :] = np.array([lam[m], omega[m], freq[m], period[m]])

            # Compute L and gm
            L = np.zeros((ndf_max, ))
            gm = 0
            for node in nodes:
                V = op.nodeEigenvector(node, m + 1)
                indf = len(op.nodeDisp(node))
                for i in range(indf):
                    Mi = op.nodeMass(node, i + 1)
                    Vi = V[i]
                    Li = Mi * Vi
                    gm += Vi**2 * Mi
                    L[i] += Li
            mode_L[m, :] = L

            # Compute MPM
            MPM = np.zeros((ndf_max, ))
            for i in range(ndf_max):
                Li = L[i]
                TMi = total_mass[i]
                MPMi = Li**2
                if gm > 0.0:
                    MPMi = MPMi / gm
                if TMi > 0.0:
                    MPMi = MPMi / TMi * 100.0
                MPM[i] = MPMi
            mode_MPM[m, :] = MPM

        # Get modal positions based on mass participation
        positions = np.argmax(mode_MPM, axis=1)
        # Take the first two, as for symmetric structures higher modes are not so important
        positions = positions[:2]

        # Calculate the first modal shape
        modalShape = np.zeros((self.data.nst, 2))
        for st in range(self.data.nst):
            if self.flag3d:
                if self.direction == 0:
                    nodetag = int(f"{nbays+1}1{st+1}")
                else:
                    nodetag = int(f"1{nbays+1}{st+1}")
            else:
                nodetag = int(f"{nbays+1}{st+1}")
                positions[0] = 0

            # First mode shape (also for 2D model)
            modalShape[st, 0] = op.nodeEigenvector(nodetag, 1, int(positions[0] + 1))
            # Second mode shape
            modalShape[st, 1] = op.nodeEigenvector(nodetag, 2, int(positions[1] + 1))

        # Normalize the modal shapes (first two modes, most likely associated with X and Y directions unless there are
        # large torsional effects)
        modalShape = np.abs(modalShape) / np.max(np.abs(modalShape), axis=0)

        # Calculate the first mode participation factor and effective modal mass
        M = np.zeros((self.data.nst, self.data.nst))
        for st in range(self.data.nst):
            M[st][st] = self.data.masses[st] / self.data.n_seismic

        # Identity matrix
        identity = np.ones((1, self.data.nst))

        gamma = np.zeros(2)
        mstar = np.zeros(2)
        for i in range(2):
            # Modal participation factor
            gamma[i] = (modalShape[:, i].transpose().dot(M)).dot(identity.transpose()) / \
                       (modalShape[:, i].transpose().dot(M)).dot(modalShape[:, i])

            # Modal mass
            mstar[i] = (modalShape[:, i].transpose().dot(M)).dot(identity.transpose())

        # Modify indices of modal properties as follows:
        # index 0 = direction x
        # index 1 = direction y
        period = np.array([period[i] for i in range(len(positions))])
        gamma = np.array([gamma[i] for i in range(len(positions))])
        mstar = np.array([mstar[i] for i in range(len(positions))])

        # Wipe analysis
        self.wipe()

        if self.flag3d:
            return period, modalShape, gamma, mstar
        else:
            return period[0], modalShape[:, 0], gamma[0], mstar[0]

    def run_single_push(self, ctrlNode, ctrlDOF, nSteps):
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

        dU = 0.1 * sum(self.data.heights) / nSteps
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

    def run_spo_algorithm(self, testType, algorithmType, nsteps, iterInit, tol):
        """Seek for a solution using different test conditions or algorithms"""

        # Set the initial values to start the while loop
        # The feature of disabling the possibility of having a negative loading has been included.
        #   adapted from a similar script by Prof. Garbaggio
        ok = 0
        step = 1
        loadf = 1.0

        # It happens so, that column shear ID matches the disp_dir ID, they are not the same thing
        col_shear_idx = self.direction + 1

        # Recording top displacement and base shear
        topDisp = np.array([op.nodeResponse(self.spo_nodes[-1], self.direction + 1, 1)])
        baseShear = np.array([0.0])
        for col in self.base_cols:
            baseShear[0] += op.eleForce(int(col), col_shear_idx)

        while step <= nsteps and ok == 0 and loadf > 0:
            ok = op.analyze(1)
            loadf = op.getTime()
            if ok != 0:
                print("[STEP] Trying relaxed convergence...")
                op.test(testType, tol * .01, int(iterInit * 50))
                ok = op.analyze(1)
                op.test(testType, tol, iterInit)
            if ok != 0:
                print("[STEP] Trying Newton with initial then current...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("Newton", "-initialThenCurrent")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                print("[STEP] Trying ModifiedNewton with initial...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("ModifiedNewton", "-initial")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                print("[STEP] Trying KrylovNewton...")
                op.test(testType, tol * .01, int(iterInit * 50))
                op.algorithm("KrylovNewton")
                ok = op.analyze(1)
                op.algorithm(algorithmType)
                op.test(testType, tol, iterInit)
            if ok != 0:
                print("[STEP] Perform a Hail Mary...")
                op.test("FixedNumIter", iterInit)
                ok = op.analyze(1)

            # Recording the displacements and base shear forces
            topDisp = np.append(topDisp, op.nodeResponse(self.spo_nodes[-1], self.direction + 1, 1))
            # topDisp1 = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 1, 1)))
            # topDisp2 = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 2, 1)))
            # topDisp = (topDisp1**2 + topDisp2**2) ** 0.5

            eleForceTemp = 0.
            for col in self.base_cols:
                eleForceTemp += op.eleForce(col, col_shear_idx)
            baseShear = np.append(baseShear, eleForceTemp)
            loadf = op.getTime()
            step += 1

        # Reverse sign of base_shear (to be positive for better visualization)
        if min(baseShear) < 0.:
            baseShear = -baseShear

        return topDisp, baseShear

    def run_spo_analysis(self, load_pattern=1, mode_shape=None):
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

        '''Load pattern definition'''
        if load_pattern == 0:
            # print("[STEP] Applying Uniform load pattern...")
            loads = [1.] * len(self.spo_nodes)
        elif load_pattern == 1:
            # print("[STEP] Applying triangular load pattern...")
            loads = np.zeros(len(self.data.heights))

            for h in range(len(self.data.heights)):
                if self.data.heights[h] != 0.:
                    loads[h] = self.data.heights[h] / sum(self.data.heights[:h + 1])

        elif load_pattern == 2:
            # Recommended
            # print("[STEP] Applying 1st mode proportional load pattern...")
            loads = mode_shape

        else:
            raise ValueError("[EXCEPTION] Wrong load pattern is supplied.")

        # Applying the load pattern
        op.timeSeries("Linear", 4)
        op.pattern("Plain", 400, 4)

        # Pushing all nodes with masses assigned to them
        nbays_x = self.data.n_bays
        nbays_y = len(self.data.spans_y)

        if self.flag3d:
            for xbay in range(1, nbays_x + 2):
                for ybay in range(1, nbays_y + 2):
                    for st in range(1, self.data.nst + 1):
                        nodepush = int(f"{xbay}{ybay}{st}")
                        fpush = loads[st-1]
                        if self.direction == 0:
                            op.load(nodepush, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
                                    self.NEGLIGIBLE, self.NEGLIGIBLE)
                        else:
                            op.load(nodepush, self.NEGLIGIBLE, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE,
                                    self.NEGLIGIBLE, self.NEGLIGIBLE)
        else:
            for fpush, nodepush in zip(loads, self.spo_nodes):
                op.load(nodepush, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
                        self.NEGLIGIBLE)

        '''Set initial analysis parameters'''
        if self.flag3d:
            op.constraints("Penalty", 1e15, 1e15)
            op.system("UmfPack")
            testType = "EnergyIncr"
        else:
            op.constraints("Plain")
            op.system("BandGeneral")
            testType = "NormDispIncr"

        op.test(testType, tol, iterInit)
        op.numberer("RCM")
        op.algorithm("KrylovNewton")
        op.integrator("DisplacementControl", self.spo_nodes[-1], self.direction + 1,
                      0.1 * sum(self.data.heights) / nsteps)
        op.analysis("Static")

        # Run the algorithm
        topDisp, baseShear = self.run_spo_algorithm(testType, "KrylovNewton", nsteps, iterInit, tol)

        # Wipe analysis
        self.wipe()

        return topDisp, baseShear

    def create_pdelta_columns(self, loads, option="EqualDOF", system="Perimeter"):
        """
        Defines pdelta columns for the 2D model
        :param loads: DataFrame                         Gravity loads over PDelta columns
        :param option: str                              Option for linking the gravity columns (Truss or EqualDOF)
        :param system: str                              System type (for now supports only Perimeter)
        :return: None
        """
        # Elastic modulus of concrete
        elastic_modulus = float((3320 * np.sqrt(self.data.fc) + 6900) * 1000 * self.fstiff)
        # Check whether Pdelta forces were provided (if not, skips step)
        if "pdelta" in loads.columns:
            # Material definition
            pdelta_mat_tag = self.data.n_bays + 2
            if system == "Perimeter":
                op.uniaxialMaterial("Elastic", pdelta_mat_tag, elastic_modulus)

            # X coordinate of the columns
            x_coord = sum(self.data.spans_x) + 3.

            # Geometric transformation for the columns
            pdelta_transf_tag = pdelta_mat_tag
            op.geomTransf("Linear", pdelta_transf_tag, 1, 0, 0)

            # Node creations and linking to the lateral load resisting structure
            zloc = 0.0
            for st in range(self.data.nst + 1):
                if st == 0:
                    node = int(f"{pdelta_mat_tag}{st}")
                    # Create and fix the node
                    op.node(node, x_coord, 0., 0.)
                    op.fix(node, 1, 1, 1, 0, 0, 0)

                else:
                    nodeFrame = int(f"{self.data.n_bays + 1}{st}")
                    node = int(f"{pdelta_mat_tag}{st}")
                    ele = int(f"1{self.data.n_bays + 1}{st}")

                    # Create the node
                    zloc += self.data.heights[st - 1]
                    op.node(node, x_coord, 0., zloc)

                    if option.lower() == "truss":
                        op.element("Truss", ele, nodeFrame, node, 5., pdelta_mat_tag)

                    elif option.lower() == "equaldof":
                        for bay in range(self.data.n_bays + 1):
                            op.equalDOF(node, int(f"{bay + 1}{st}"), 1)

                    else:
                        raise ValueError("[EXCEPTION] Wrong option for linking gravity columns "
                                         "(needs to be Truss or EqualDOF")

            # Fix out-of-plane (gravity columns are necessary when the whole building is not modelled)
            op.fixY(0.0, 0, 1, 0, 0, 0, 1)

            # Creation of P-Delta column elements
            agcol = 0.5**2
            izgcol = 0.5**4 / 12 / 1.0e4
            for st in range(1, self.data.nst + 1):
                eleid = int(f"2{pdelta_mat_tag}{st}")
                node_i = int(f"{pdelta_mat_tag}{st-1}")
                node_j = int(f"{pdelta_mat_tag}{st}")
                op.element("elasticBeamColumn", eleid, node_i, node_j, agcol, elastic_modulus, self.NEGLIGIBLE,
                           self.NEGLIGIBLE, self.NEGLIGIBLE, izgcol, pdelta_transf_tag)

            # Definition of loads
            op.timeSeries("Linear", 11)
            op.pattern("Plain", 11, 11)
            pdelta_loads = loads["pdelta"].reset_index(drop=True)
            for idx in pdelta_loads.index:
                load = pdelta_loads.loc[idx]
                if not load:
                    pass
                else:
                    op.load(int(f"{pdelta_mat_tag}{idx + 1}"), self.NEGLIGIBLE, self.NEGLIGIBLE, -load, self.NEGLIGIBLE,
                            self.NEGLIGIBLE, self.NEGLIGIBLE)


if __name__ == "__main__":

    from pathlib import Path
    from src.input import Input
    import pandas as pd
    import pickle
    import sys

    directory = Path.cwd().parents[1] / ".applications/LOSS Validation Manuscript/space/case4"

    # actionx = directory.parents[0] / "sample" / "actionx.csv"
    # actiony = directory.parents[0] / "sample" / "actiony.csv"
    # csx = directory / "Cache/solution_cache_space_x.csv"
    # csy = directory / "Cache/solution_cache_space_y.csv"
    # csg = directory / "Cache/solution_cache_space_gr.csv"
    input_file = directory / "ipbsd_input.csv"
    # hinge_models = Path.cwd().parents[0] / "tempHinge.pickle"
    direction = 0
    modalShape = np.array([.58, 1.0])

    # # Read the cross-section files
    # csx = pd.read_csv(csx, index_col=0).iloc[540]
    # csy = pd.read_csv(csy, index_col=0).iloc[540]
    # csg = pd.read_csv(csg, index_col=0).iloc[540]

    # csy["hi1"] = 0.6
    # csy["h1"] = 0.8
    # csy["hi2"] = 0.6
    # csy["h2"] = 0.8
    # csy["hi3"] = 0.55
    # csy["h3"] = 0.75
    # csy["hi4"] = 0.55
    # csy["h4"] = 0.75
    #
    # csg["hy1"] = 0.8
    # csg["hy2"] = 0.8
    # csg["hy3"] = 0.75
    # csg["hy4"] = 0.75

    # cs = {"x_seismic": csx, "y_seismic": csy, "gravity": csg}

    # actionx = pd.read_csv(actionx)
    # actiony = pd.read_csv(actiony)

    lat_action = [267, 465, 633, 628]

    # Hinge models
    with open(Path.cwd().parents[1] / "tests/hinge_temp.pickle", 'rb') as file:
        data = pickle.load(file)

    with open(Path.cwd().parents[1] / "tests/sol_temp.pickle", 'rb') as file:
        cs = pickle.load(file)

    hinge = data

    hinge_elastic = {"x_seismic": None, "y_seismic": None, "gravity": None}
    fstiff = 0.5

    # Read input data
    data = Input()
    data.read_inputs(input_file)

    # analysis = OpenSeesRun3D(data, cs, hinge=hinge_elastic, direction=direction, fstiff=fstiff, system="space",
    #                          pflag=True)
    # results = analysis.elastic_analysis_3d(analysis=3, lat_action=lat_action, grav_loads=None)

    # ma = OpenSeesRun3D(data, cs, hinge=hinge_elastic, direction=direction, fstiff=fstiff, system="space")
    # ma.create_model(True)
    # ma.define_masses()
    # model_periods, modalShape, gamma, mstar = ma.ma_analysis(2)
    # ma.wipe()

    spo = OpenSeesRun(data, cs, fstiff, hinge=hinge, direction=direction, system="space")
    spo.create_model(gravity=True)
    spo.define_masses()
    topDisp, baseShear = spo.run_spo_analysis(load_pattern=2, mode_shape=modalShape)
    spo.wipe()

    spo_results = {"d": topDisp, "v": baseShear}
    with open(directory / f"temp_spo.pickle", 'wb') as handle:
        pickle.dump(spo_results, handle)
