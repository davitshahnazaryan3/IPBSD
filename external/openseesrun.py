"""
Runs OpenSees software for structural analysis (in future may be changed with anastruct or other software)
For now only ELFM, pushover to be added later if necessary.
The idea is to keep it as simple as possible, and with fewer runs of this class.
This is  linear elastic analysis in order to obtain demands on structural components for further detailing
"""
import openseespy.opensees as op
import numpy as np


class OpenSeesRun:
    def __init__(self, i_d, cs, fstiff=0.5, hinge=None, pflag=False):
        """
        initializes model creation for analysing
        :param i_d: dict                        Provided input arguments for the framework
        :param cs: DataFrame                    Cross-sections of the solution
        :param fstiff: float                    Stiffness reduction factor
        :param hinge: DataFrame                 Idealized plastic hinge model parameters
        :param pflag: bool                      Print info
        """
        self.i_d = i_d
        self.cs = cs
        self.fstiff = fstiff
        self.BEAM_TRANSF_TAG = 1
        self.COL_TRANSF_TAG = 2
        self.NEGLIGIBLE = 1.e-10
        self.pflag = pflag
        self.hinge = hinge
        # Base nodes for SPO recorder
        self.base_nodes = []
        # Base columns (base node reaction recorders seem to be not working)
        self.base_cols = []
        # Rightmost nodes for pushover load application
        self.spo_nodes = []

    @staticmethod
    def wipe():
        """
        wipes model
        :return: None
        """
        op.wipe()

    def create_model(self, mode='3D'):
        """
        creates the model
        :param mode: str                        2D or 3D
        :return: array                          Beam and column element tags
        """
        self.wipe()
        # For now, only 3D mode is supported
        if mode == '3D':
            op.model('Basic', '-ndm', 3, '-ndf', 6)
        elif mode == '2D':
            op.model('Basic', '-ndm', 2, '-ndf', 3)
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
        op.geomTransf(col_transf_type, self.BEAM_TRANSF_TAG, 0, 1, 0)
        op.geomTransf(col_transf_type, self.COL_TRANSF_TAG, 0, 1, 0)

    def define_nodes(self, fix_out_of_plane=True):
        """
        defines nodes and fixities
        :param fix_out_of_plane: bool           If 3D space is used for a 2D model then True
        :return: None
        """
        xloc = 0.
        for bay in range(0, int(self.i_d.n_bays + 1)):
            zloc = 0.
            nodetag = int(str(1 + bay) + '0')
            self.base_nodes.append(nodetag)
            op.node(nodetag, xloc, 0, zloc)
            op.fix(nodetag, 1, 1, 1, 1, 1, 1)
            for st in range(1, self.i_d.nst + 1):
                zloc += self.i_d.heights[st - 1]
                nodetag = int(str(1 + bay) + str(st))
                op.node(nodetag, xloc, 0, zloc)
                if bay == self.i_d.n_bays:
                    self.spo_nodes.append(nodetag)

            if bay == int(self.i_d.n_bays):
                pass
            else:
                xloc += self.i_d.spans_x[bay]
        if fix_out_of_plane:
            op.fixY(0.0, 0, 1, 0, 0, 0, 1)
        if self.pflag:
            print("[NODE] Nodes created!")

    def define_masses(self):
        """
        Define masses
        :return: None
        """
        for st in range(self.i_d.nst):
            for bay in range(self.i_d.n_bays+1):
                if bay == 0 or bay == self.i_d.n_bays:
                    mass = self.i_d.masses[st] / (2*self.i_d.n_bays) / self.i_d.n_seismic
                else:
                    mass = self.i_d.masses[st] / self.i_d.n_bays / self.i_d.n_seismic
                op.mass(int(f"{bay+1}{st+1}"), mass, self.NEGLIGIBLE, mass, self.NEGLIGIBLE, self.NEGLIGIBLE,
                        self.NEGLIGIBLE)

    def ma_analysis(self, num_modes):
        """
        Runs modal analysis
        :param num_modes: DataFrame                 Design solution, cross-section dimensions
        :return: list                               Modal periods
        """
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
        modalShape = np.zeros(self.i_d.nst)
        for st in range(self.i_d.nst):
            modalShape[st] = op.nodeEigenvector(int(f"{self.i_d.n_bays+1}{st+1}"), 1, 1)

        # Normalize the modal shapes
        modalShape = modalShape / max(modalShape, key=abs)

        # Calculate the first mode participation factor and effective modal mass
        M = np.zeros((self.i_d.nst, self.i_d.nst))
        for st in range(self.i_d.nst):
            M[st][st] = self.i_d.masses[st] / self.i_d.n_seismic
        identity = np.ones((1, self.i_d.nst))
        gamma = (modalShape.transpose().dot(M)).dot(identity.transpose()) / \
                (modalShape.transpose().dot(M)).dot(modalShape)
        mstar = (modalShape.transpose().dot(M)).dot(identity.transpose())

        return period, modalShape, float(gamma), float(mstar)

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
        eiz = elastic_modulus * iz

        # Curvatures at yield
        phiyNeg = phiyPos = my / eiz

        # Material model creation
        hingeMTag1 = int(f"101{et}")
        hingeMTag2 = int(f"102{et}")

        myNeg = myPos = my
        mpNeg = mpPos = ap * my
        muNeg = muPos = r * my

        phipNeg = phipPos = phiyNeg * mu_phi
        phiuNeg = phiuPos = phipNeg + (mpNeg - muNeg) / (app * my / phiyNeg)

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
        op.section("Elastic", int_tag, elastic_modulus, area, iz, iz, 0.4 * elastic_modulus, 0.01)
        op.section("Uniaxial", ph_tag1, hingeMTag1, 'Mz')
        op.section("Uniaxial", ph_tag2, hingeMTag2, 'Mz')

        op.beamIntegration("HingeRadau", integration_tag, ph_tag1, lp, ph_tag2, lp, int_tag)
        op.element("forceBeamColumn", et, inode, jnode, gt, integration_tag)

    def create_elements(self):
        """
        creates elements
        consideration given only for ELFM, so capacities are arbitrary
        :return: list                               Element tags
        """
        n_beams = self.i_d.nst * self.i_d.n_bays
        n_cols = self.i_d.nst * (self.i_d.n_bays + 1)
        capacities_beam = [5000.0] * n_beams
        capacities_col = [5000.0] * n_cols
        # beam generation
        beam_id = 0
        beams = []
        for bay in range(1, int(self.i_d.n_bays + 1)):
            for st in range(1, int(self.i_d.nst + 1)):
                # Read hinge model if provided
                if self.hinge is not None:
                    eleHinge = self.hinge[(self.hinge["Element"] == "Beam") & (self.hinge["Bay"] == bay) & (
                                self.hinge["Storey"] == st)].reset_index(drop=True)
                else:
                    eleHinge = None

                # Parameters for elastic static analysis
                next_bay = bay + 1
                b_beam = self.cs[f'b{st}']
                h_beam = self.cs[f'b{st}']
                lp = 1.0 * h_beam  # not important for linear static analysis
                my = capacities_beam[beam_id]
                beam_id += 1
                et = int(f"1{bay}{st}")
                gt = self.BEAM_TRANSF_TAG
                inode = int(f"{bay}{st}")
                jnode = int(f"{next_bay}{st}")
                self.lumped_hinge_element(et, gt, inode, jnode, my, lp, self.i_d.fc, b_beam, h_beam,
                                          hingeModel=eleHinge)
                # Record beam element tag
                beams.append(et)
        if self.pflag:
            print("[ELEMENT] Beams created!")

        # column generation
        col_id = 0
        columns = []
        for bay in range(1, int(self.i_d.n_bays + 2)):
            for st in range(1, int(self.i_d.nst + 1)):
                # Read hinge model if provided
                if self.hinge is not None:
                    eleHinge = self.hinge[(self.hinge["Element"] == "Column") & (self.hinge["Bay"] == bay) & (
                                self.hinge["Storey"] == st)].reset_index(drop=True)
                else:
                    eleHinge = None

                # Parameters for elastic static analysis
                previous_st = st - 1
                my = capacities_col[col_id]
                col_id += 1
                if bay == 1 or bay == 1 + int(self.i_d.n_bays):
                    b_col = h_col = self.cs[f'he{st}']
                else:
                    b_col = h_col = self.cs[f'hi{st}']
                lp = h_col * 1.0  # not important for linear static analysis
                et = int(f"2{bay}{st}")
                gt = self.COL_TRANSF_TAG
                inode = int(f"{bay}{previous_st}")
                jnode = int(f"{bay}{st}")
                self.lumped_hinge_element(et, gt, inode, jnode, my, lp, self.i_d.fc, b_col, h_col, hingeModel=eleHinge)
                columns.append(et)
                # Base columns
                if st == 1:
                    self.base_cols.append(int(f"2{bay}{st}"))

        if self.pflag:
            print("[ELEMENT] Columns created!")
        return beams, columns

    def pdelta_columns(self, loads, option="EqualDOF", system="Perimeter"):
        """
        Defines pdelta columns
        :param loads: DataFrame                         Gravity loads over PDelta columns
        :param option: str                              Option for linking the gravity columns (Truss or EqualDOF)
        :param system: str                              System type (for now supports only Perimeter)
        :return: None
        """
        # Elastic modulus of concrete
        elastic_modulus = (3320 * np.sqrt(self.i_d.fc) + 6900) * 1000 * self.fstiff
        # Check whether Pdelta forces were provided (if not, skips step)
        if "pdelta" in loads.columns:
            # Material definition
            pdelta_mat_tag = self.i_d.n_bays + 2
            if system == "Perimeter":
                op.uniaxialMaterial("Elastic", pdelta_mat_tag, elastic_modulus)

            # X coordinate of the columns
            x_coord = sum(self.i_d.spans_x) + 3.

            # Geometric transformation for the columns
            pdelta_transf_tag = pdelta_mat_tag
            op.geomTransf("Linear", pdelta_transf_tag, 1, 0, 0)

            # Node creations and linking to the lateral load resisting structure
            zloc = 0.0
            for st in range(self.i_d.nst+1):
                if st == 0:
                    node = int(f"{pdelta_mat_tag}{st}")
                    # Create and fix the node
                    op.node(node, x_coord, 0., 0.)
                    op.fix(node, 1, 1, 1, 0, 0, 0)

                else:
                    nodeFrame = int(f"{self.i_d.n_bays + 1}{st}")
                    node = int(f"{pdelta_mat_tag}{st}")
                    ele = int(f"1{self.i_d.n_bays + 1}{st}")

                    # Create the node
                    zloc += self.i_d.heights[st - 1]
                    op.node(node, x_coord, 0., zloc)

                    if option.lower() == "truss":
                        op.element("Truss", ele, nodeFrame, node, 5., pdelta_mat_tag)

                    elif option == "EqualDOF":
                        for bay in range(self.i_d.n_bays+1):
                            op.equalDOF(node, int(f"{bay + 1}{st}"), 1)

                    else:
                        raise ValueError("[EXCEPTION] Wrong option for linking gravity columns "
                                         "(needs to be Truss or EqualDOF")

            # Fix out-of-plane (gravity columns are necessary when the whole building is not modelled)
            op.fixY(0.0, 0, 1, 0, 0, 0, 1)

            # Creation of P-Delta column elements
            agcol = 2.**2
            izgcol = 2.**4/12
            young_modulus = float(elastic_modulus)
            for st in range(1, self.i_d.nst + 1):
                eleid = int(f"2{pdelta_mat_tag}{st}")
                node_i = int(f"{pdelta_mat_tag}{st-1}")
                node_j = int(f"{pdelta_mat_tag}{st}")
                op.element("elasticBeamColumn", eleid, node_i, node_j, agcol, young_modulus, self.NEGLIGIBLE,
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

    def gravity_loads(self, action, elements):
        """
        Defines gravity loads (only distributed, code for point loads to be here)
        :param action: list                         Acting gravity loads
        :param elements: list                       Element IDs
        :return: None
        """
        op.timeSeries("Linear", 2)
        op.pattern("Plain", 2, 2)
        for ele in elements:
            storey = int(str(ele)[-1]) - 1
            op.eleLoad('-ele', ele, '-type', '-beamUniform', action[storey], self.NEGLIGIBLE)

    def elfm_loads(self, action):
        """
        applies lateral loads
        :param action: list                         Acting lateral loads
        :return: None
        """
        op.timeSeries("Linear", 1)
        op.pattern("Plain", 1, 1)
        for st in range(1, int(self.i_d.nst + 1)):
            for bay in range(1, int(self.i_d.n_bays + 2)):
                op.load(int(f"{bay}{st}"), action[st - 1] / (self.i_d.n_bays + 1), 0, 0, 0, 0, 0)

    def static_analysis(self):
        """
        carries out static ELFM analysis
        :return: None
        """
        op.constraints("Plain")
        op.numberer("Plain")
        op.system("BandGeneral")
        op.test("NormDispIncr", 1.0e-8, 6)
        op.integrator("LoadControl", 0.1)
        op.algorithm("Newton")
        op.analysis("Static")
        op.analyze(10)
        op.loadConst("-time", 0.0)

    def spo_analysis(self, load_pattern=2, mode_shape=None):
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
        iterInit = 50
        testType = "NormDispIncr"
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
                    loads.append(self.i_d.heights[h]/sum(self.i_d.heights[:h]))
        elif load_pattern == 2:
            # print("[STEP] Applying 1st mode proportional load pattern...")
            for i in mode_shape:
                loads.append(i)
        else:
            raise ValueError("[EXCEPTION] Wrong load pattern is supplied.")

        # Applying the load pattern
        op.timeSeries("Linear", 4)
        op.pattern("Plain", 400, 4)
        for fpush, nodepush in zip(loads, self.spo_nodes):
            op.load(nodepush, fpush, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE, self.NEGLIGIBLE,
                    self.NEGLIGIBLE)

        '''Set initial analysis parameters'''
        op.constraints("Plain")
        op.numberer("RCM")
        op.system("BandGeneral")
        op.test(testType, tol, iterInit, 0)
        op.algorithm(algorithmType)
        op.integrator("DisplacementControl", self.spo_nodes[-1], 1, 1./nsteps)
        op.analysis("Static")

        '''Seek for a solution using different test conditions or algorithms'''
        # Set the initial values to start the while loop
        # The feature of disabling the possibility of having a negative loading has been included.
        #   adapted from a similar script by Prof. Garbaggio
        ok = 0
        step = 1
        loadf = 1.0

        # Recording top displacement and base shear
        topDisp = np.array([op.nodeResponse(self.spo_nodes[-1], 1, 1)])
        baseShear = np.array([0.0])
        for col in self.base_cols:
            baseShear[0] += op.eleForce(col, 1, 3)

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
            topDisp = np.append(topDisp, (op.nodeResponse(self.spo_nodes[-1], 1, 1)))
            eleForceTemp = 0.
            for col in self.base_cols:
                eleForceTemp += op.eleForce(col, 1, 3)
            baseShear = np.append(baseShear, eleForceTemp)

            loadf = op.getTime()
            step += 1

        # Reverse sign of base_shear
        if min(baseShear) < 0.:
            baseShear = -baseShear

        return topDisp, baseShear

    def define_recorders(self, beams, columns, analysis):
        """
        recording results
        :param beams: list                          Beam element tags
        :param columns: list                        Column element tags
        :param analysis: int                        Analysis type
        :return: ndarray                            Demands on beams and columns
        """
        if analysis in [1, 2, 3]:
            b = np.zeros((self.i_d.nst, self.i_d.n_bays))
            c = np.zeros((self.i_d.nst, self.i_d.n_bays + 1))

            results = {"Beams": {"M": {"Pos": b.copy(), "Neg": b.copy()}, "N": b.copy(), "V": b.copy()},
                       "Columns": {"M": c.copy(), "N": c.copy(), "V": c.copy()}}
            # Beams, counting: bottom to top, left to right
            ele = 0
            for bay in range(self.i_d.n_bays):
                for st in range(self.i_d.nst):
                    results["Beams"]["M"]["Pos"][st][bay] = abs(op.eleForce(beams[ele], 11))
                    results["Beams"]["M"]["Neg"][st][bay] = abs(op.eleForce(beams[ele], 5))
                    results["Beams"]["N"][st][bay] = max(op.eleForce(beams[ele], 1),
                                                         op.eleForce(beams[ele], 7), key=abs)
                    results["Beams"]["V"][st][bay] = max(abs(op.eleForce(beams[ele], 3)),
                                                         abs(op.eleForce(beams[ele], 9)))
                    ele += 1

            # Columns
            ele = 0
            for bay in range(self.i_d.n_bays + 1):
                for st in range(self.i_d.nst):
                    results["Columns"]["M"][st][bay] = max(abs(op.eleForce(columns[ele], 5)),
                                                           abs(op.eleForce(columns[ele], 11)))
                    # Negative N is tension, Positive N is compression
                    results["Columns"]["N"][st][bay] = max(op.eleForce(columns[ele], 3),
                                                           op.eleForce(columns[ele], 9), key=abs)
                    results["Columns"]["V"][st][bay] = max(abs(op.eleForce(columns[ele], 1)),
                                                           abs(op.eleForce(columns[ele], 7)))
                    ele += 1
        else:
            # TODO, fix recording when applying RMSA, analysis type if condition seems incorrect
            n_beams = self.i_d.nst * self.i_d.n_bays
            n_cols = self.i_d.nst * (self.i_d.n_bays + 1)
            results = {"Beams": {}, "Columns": {}}
            if analysis != 4 and analysis != 5:
                for i in range(n_beams):
                    results["Beams"][i] = {
                        "M": abs(max(op.eleForce(beams[i], 5), op.eleForce(beams[i], 11), key=abs)),
                        "N": abs(max(op.eleForce(beams[i], 1), op.eleForce(beams[i], 7), key=abs)),
                        "V": abs(max(op.eleForce(beams[i], 3), op.eleForce(beams[i], 9), key=abs))}
                for i in range(n_cols):
                    results["Columns"][i] = {
                        "M": abs(max(op.eleForce(columns[i], 5), op.eleForce(columns[i], 11), key=abs)),
                        "N": abs(max(op.eleForce(columns[i], 3), op.eleForce(columns[i], 9), key=abs)),
                        "V": abs(max(op.eleForce(columns[i], 1), op.eleForce(columns[i], 7), key=abs))}
            else:
                for i in range(n_beams):
                    results["Beams"][i] = {"M": np.array([op.eleForce(beams[i], 5), op.eleForce(beams[i], 11)]),
                                           "N": np.array([op.eleForce(beams[i], 1), op.eleForce(beams[i], 7)]),
                                           "V": np.array([op.eleForce(beams[i], 3), op.eleForce(beams[i], 9)])}
                for i in range(n_cols):
                    results["Columns"][i] = {"M": np.array([op.eleForce(columns[i], 5), op.eleForce(columns[i], 11)]),
                                             "N": np.array([op.eleForce(columns[i], 3), op.eleForce(columns[i], 9)]),
                                             "V": np.array([op.eleForce(columns[i], 1), op.eleForce(columns[i], 7)])}

        return results


if __name__ == "__main__":
    from client.master import Master
    from pathlib import Path

    directory = Path.cwd().parents[1]

    outputPath = directory / ".applications/case1/Output"

    csd = Master(directory)
    input_file = directory / ".applications/case1/ipbsd_input.csv"

    csd.read_input(input_file, "Hazard-LAquila-Soil-C.pkl", outputPath=outputPath)

    cs = {'he1': 0.35, 'hi1': 0.4, 'b1': 0.25, 'h1': 0.45, 'he2': 0.3, 'hi2': 0.35, 'b2': 0.25,
          'h2': 0.45, 'he3': 0.25, 'hi3': 0.3, 'b3': 0.25, 'h3': 0.45, 'T': 0.936}
    analysis = 3
    op = OpenSeesRun(csd.data, cs, analysis=analysis, fstiff=1.0)
    beams, columns = op.create_model()
    action = [160, 200, 200]
    gravity = [16.2, 13.5, 13.5]
    op.gravity_loads(gravity, beams)
    op.elfm_loads(action)
    op.static_analysis()
    response = op.define_recorders(beams, columns, analysis=analysis)
    print(response)
