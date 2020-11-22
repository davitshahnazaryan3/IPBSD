import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq


class Iterations:
    def __init__(self, ipbsd, sols, spo_file, target_MAFC, analysis_type, damping, num_modes, fstiff, rebar_cover,
                 outputPath):
        """
        Initialize
        """
        self.ipbsd = ipbsd
        self.sols = sols
        self.spo_file = spo_file
        self.target_MAFC = target_MAFC
        self.analysis_type = analysis_type
        self.damping = damping
        self.num_modes = num_modes
        self.fstiff = fstiff
        self.rebar_cover = rebar_cover
        self.spo_data = None
        self.outputPath = outputPath
        self.omegaWarn = False
        self.spo_validate = True

    def iterate_phase_3(self, opt_sol, omega, read=True):
        """
        Runs phase 3 of the framework
        :param opt_sol: Series                          Solution containing c-s and modal properties
        :param omega: float                             Overstrength ratio
        :param read: bool                               Whether to read the input file or not
        :return part_factor: float                      Participation factor of first mode
        :return m_star: float                           Effective first modal mass
        :return say: float                              Spectral acceleration at yield in g
        :return dy: float                               Spectral displacement at yield in m
        :return spo2ida_data: dict                      SPO2IDA results
        """
        """Perform SPO2IDA"""
        """Estimate parameters for SPO curve and compare with assumed shape"""
        period = round(float(opt_sol['T']), 1)
        if read:
            self.spo_data = self.ipbsd.data.initial_spo_data(period, self.spo_file)
        spo2ida_data = self.ipbsd.perform_spo2ida(self.spo_data)
        print("[SUCCESS] SPO2IDA was performed")

        """Yield strength optimization for MAFC and verification"""
        part_factor = opt_sol["Part Factor"]
        m_star = opt_sol["Mstar"]
        say, dy = self.ipbsd.verify_mafc(period, spo2ida_data, part_factor, self.target_MAFC, omega, hazard="True")
        print("[SUCCESS] MAFC was validated")
        return part_factor, m_star, say, dy, spo2ida_data

    def iterate_phase_4(self, say, dy, sa, period_range, solution, modes, table_sls, t_lower, t_upper):
        """
        Runs phase 4 of the framework
        :param say: float                               Spectral acceleration at yield in g
        :param dy: float                                Spectral displacement at yield in m
        :param sa: list                                 Spectral accelerations in g of the spectrum
        :param period_range: list                       Periods of the spectrum
        :param solution: Series                         Solution containing c-s and modal properties
        :param modes: dict                              Periods and normalized modal shapes of the solution
        :param table_sls: DataFrame                     Table with SLS parameters
        :param t_lower: float                           Lower period limit
        :param t_upper: float                           Upper period limit
        :return forces: DataFrame                       Acting forces
        :return demands: dict                           Demands on the structural elements
        :return details: dict                           Designed element properties from the moment-curvature
                                                        relationship
        :return hard_ductility: float                   Estimated system hardening ductility
        :return fract_ductility: float                  Estimated system fracturing ductility
        :return warn: bool                              Indicates whether any warnings were displayed
        :return warnings: dict                          Dictionary of boolean warnings for each structural element
                                                        i.e. local ductility requirements not met in Detailing
        """
        if self.analysis_type == 4 or self.analysis_type == 5:
            se_rmsa = self.ipbsd.get_sa_at_period(say, sa, period_range, modes["Periods"])
        else:
            se_rmsa = None

        self.num_modes = min(self.num_modes, self.ipbsd.data.nst)
        if self.analysis_type == 4 or self.analysis_type == 5:
            corr = self.ipbsd.get_correlation_matrix(modes["Periods"], self.num_modes, damping=self.damping)
        else:
            corr = None
        forces = self.ipbsd.get_action(solution, say, pd.DataFrame.from_dict(table_sls), self.ipbsd.data.w_seismic,
                                       self.analysis_type, self.num_modes, modes, modal_sa=se_rmsa)

        print("[SUCCESS] Actions on the structure for analysis were estimated")
        # if self.export_cache:
        #     self.export_results(self.outputPath / "Cache/action", forces, "csv")
        yield forces

        """Perform ELF analysis"""
        if self.analysis_type == 1:
            demands = self.ipbsd.run_muto_approach(solution, list(forces["Fi"]), self.ipbsd.data.h,
                                                   self.ipbsd.data.spans_x)
        elif self.analysis_type == 2:
            demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), fstiff=self.fstiff)
        elif self.analysis_type == 3:
            demands = self.ipbsd.run_analysis(self.analysis_type, solution, list(forces["Fi"]), list(forces["G"]),
                                              fstiff=self.fstiff)
        elif self.analysis_type == 4 or self.analysis_type == 5:
            demands = {}
            for mode in range(self.num_modes):
                demands[f"Mode{mode + 1}"] = self.ipbsd.run_analysis(self.analysis_type, solution,
                                                                     list(forces["Fi"][mode]), fstiff=self.fstiff)
            demands = self.ipbsd.perform_cqc(corr, demands)
            if self.analysis_type == 5:
                demands_gravity = self.ipbsd.run_analysis(self.analysis_type, solution, grav_loads=list(forces["G"]),
                                                          fstiff=self.fstiff)
                # Combining gravity and RSMA results
                for eleType in demands_gravity.keys():
                    for dem in demands_gravity[eleType].keys():
                        demands[eleType][dem] = demands[eleType][dem] + demands_gravity[eleType][dem]

        else:
            raise ValueError("[EXCEPTION] Incorrect analysis type...")

        print("[SUCCESS] Analysis completed and demands on structural elements were estimated.")
        # if self.export_cache:
        #     self.export_results(self.outputPath / "Cache/demands", demands, "pickle")
        yield demands
        # todo, estimation of global peak to yield ratio to be added
        """Design the structural elements"""
        details, hinge_models, mu_c, mu_f, warn, warnings = self.ipbsd.design_elements(demands, solution, modes,
                                                                                       t_lower, t_upper, dy,
                                                                                       fstiff=self.fstiff,
                                                                                       cover=self.rebar_cover)

        print("[SUCCESS] Section detailing done. Element idealized Moment-Curvature relationships obtained")

        yield details, hinge_models, mu_c, mu_f
        yield warn, warnings

    def derive_spo_shape(self, spo):
        """
        Fits a curve to the model SPO shape
        :param spo: dict                            Top displacement and base shear
        :return: ndarrays                           Fitted top displacement and base shear
        """
        # Top displacement and base shear
        x = spo[0]
        y = spo[1]

        def getIndex(My, data):
            if np.where(data >= My)[0].size == 0:
                return np.nan
            else:
                return np.where(data >= My)[0][0]

        def getEquation(p1, p2):
            points = [p1, p2]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]
            return m, c

        # Get maximum moment point
        Vmax = max(y)
        dmax = x[getIndex(Vmax, y)]

        # Get initial stiffness
        m1 = 0.2 * Vmax
        d1 = x[getIndex(m1, y)]
        stiff_elastic = m1 / d1
        temp = y / x
        stfIdx = np.where(temp < 0.9 * stiff_elastic)[0][0]
        d2 = x[stfIdx]
        m2 = y[stfIdx]

        # Find point of plasticity initiation
        stf0 = (y[1] - y[0]) / (x[1] - x[0])
        for i in range(1, len(x)):
            stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            if stf1 <= 0.5 * stf0:
                break
            else:
                stf0 = stf1

        if i == getIndex(Vmax, y):
            i = i - 5

        dPl = x[i]
        mPl = y[i]

        a0, b0 = getEquation((d1, m1), (d2, m2))
        a1, b1 = getEquation((dPl, mPl), (dmax, Vmax))

        # Find intersection point, i.e. the nominal yield point
        xint = (b1 - b0) / (a0 - a1)
        yint = a0 * xint + b0

        # Now, identify the residual strength point (here defined at V=0)
        xlast = max(x)
        ylast = y[-1]

        # Note: might need to reduce xlast slightly, as the curve shape might be steeper at earlier levels
        xres = (Vmax * xlast - ylast * dmax) / (Vmax - ylast)

        # Get the curve
        d = np.array([0., xint, dmax, xres])
        v = np.array([0., yint, Vmax, 1e-9])

        return d, v

    def spo2ida_parameters(self, d, v, t):
        """
        Updates the SPO2IDA parameters
        :param d: ndarray                       Displacements
        :param v: ndarray                       Base shear forces
        :param t: float                         Fundamental period
        :return: dict                           Updated SPO2IDA input
        """
        # Residual strength
        r = v[-1]
        # Hardening ductility
        muC = d[2] / d[1]
        # Fracturing ductility
        muF = d[3] / d[1]
        # Hardening slope
        a = (v[2] / v[1] - 1.) / (muC - 1.)
        # Softening slope
        ap = (r - v[2] / v[1]) / (muF - muC)
        # Pinch weight
        pw = 1.0
        # Create a Dictionary
        spo_data = {"mc": muC, "a": a, "ac": ap, "r": r, "mf": muF, "pw": pw, "T": t}
        return spo_data

    def compare_value(self, x, y, tol=0.05):
        """
        Verify whether x is within tolerance bounds of y
        :param x: float
        :param y: float
        :param tol: float
        :return: bool
        """
        if y - tol * y <= x <= y + tol * y:
            return True
        else:
            return False

    def seek_solution(self, warnings, opt):
        """
        Seeks a solution within the already generated section combinations file if any warnings were recorded
        :param warnings: dict                       Dictionary of boolean warnings for each structural element
                                                    For any warning cross-section dimensions will be modified
        :param opt: Series                          Optimal solution
        :return: Series                             Solution containing c-s and modal properties
        :return: dict                               Modes corresponding to the solution for RSMA
        """
        nst = self.ipbsd.data.nst
        # Remove column values
        cols_to_drop = ["T", "Weight", "Mstar", "Part Factor"]
        opt.loc[cols_to_drop] = np.nan

        # After modifications, equally between c-s of two storeys might not hold
        # Increment for cross-section modifications for elements with warnings
        increment = 0.05
        any_warnings = 0
        for ele in warnings["Columns"]:
            if warnings["Columns"][ele] == 1:
                # Increase section cross-section
                storey = ele[1]
                bay = int(ele[3])
                if bay == 1:
                    opt[f"he{storey}"] = opt[f"he{storey}"] + increment

                else:
                    opt[f"hi{storey}"] = opt[f"hi{storey}"] + increment
                any_warnings = 1

        for ele in warnings["Beams"]:
            storey = ele[1]
            if warnings["Beams"][ele] == 1:
                # Increase section cross-section
                opt[f"b{storey}"] = opt[f"he{storey}"]
                opt[f"h{storey}"] = opt[f"h{storey}"] + increment
                any_warnings = 1

        if any_warnings == 1:
            '''Enforce constraints'''
            # Column storey constraints
            for st in range(1, nst, 2):
                if opt[f"he{st}"] != opt[f"he{st + 1}"]:
                    opt[f"he{st + 1}"] = opt[f"he{st}"]
                if opt[f"he{st}"] < opt[f"he{st + 1}"]:
                    opt[f"he{st}"] = opt[f"he{st + 1}"]

                if opt[f"hi{st}"] != opt[f"hi{st + 1}"]:
                    opt[f"hi{st + 1}"] = opt[f"hi{st}"]
                if opt[f"hi{st}"] < opt[f"hi{st + 1}"]:
                    opt[f"hi{st}"] = opt[f"hi{st + 1}"]

            # Column bay constraints
            for st in range(1, nst + 1):
                if opt[f"hi{st}"] > opt[f"he{st}"] + 0.2:
                    opt[f"he{st}"] = opt[f"hi{st}"] - 0.2

                if opt[f"hi{st}"] < opt[f"he{st}"]:
                    opt[f"hi{st}"] = opt[f"he{st}"]

            # Beam width and external column width constraint
            if opt[f"b{nst}"] != opt[f"he{nst}"]:
                opt[f"b{nst}"] = opt[f"he{nst}"] = max(opt[f"b{nst}"], opt[f"he{nst}"])

            # Beam equality constraint
            for st in range(1, nst, 2):
                if opt[f"b{st}"] != opt[f"b{st + 1}"]:
                    opt[f"b{st}"] = opt[f"b{st + 1}"] = max(opt[f"b{st}"], opt[f"b{st + 1}"])

                if opt[f"h{st}"] != opt[f"h{st + 1}"]:
                    opt[f"h{st}"] = opt[f"h{st + 1}"] = max(opt[f"h{st}"], opt[f"h{st + 1}"])

            # Max value limits
            for st in range(1, nst + 1):
                if opt[f"b{st}"] > 0.55:
                    opt[f"b{st}"] = 0.55
                if opt[f"h{st}"] > 0.75:
                    opt[f"h{st}"] = 0.75
                if opt[f"hi{st}"] > 0.75:
                    opt[f"hi{st}"] = 0.75
                if opt[f"he{st}"] > 0.75:
                    opt[f"he{st}"] = 0.75

            # Beam width to height ratio constraint
            for st in range(1, nst):
                if opt[f"b{st}"] > opt[f"h{st}"] - 0.1:
                    opt[f"h{st}"] = opt[f"b{st}"] + 0.1
                if opt[f"b{st}"] < opt[f"h{st}"] - 0.3:
                    opt[f"b{st}"] = opt[f"b{st}"] - 0.3

        # Finding a matching solution from the already generated DataFrame
        solution = self.sols[self.sols == opt].dropna(thresh=len(self.sols.columns) - 4)
        solution = self.sols.loc[solution.index]

        if solution.empty:
            raise ValueError("[EXCEPTION] No solution satisfying the period range condition was found!")
        else:
            solution, modes = self.ipbsd.get_all_section_combinations(t_lower=None, t_upper=None,
                                                                      solution=solution.iloc[0], data=self.ipbsd.data,
                                                                      cache_dir=self.outputPath / "Cache")
            return solution, modes

    def run_model(self, opt_sol, hinge_models, forces, say, omega, t_upper):
        """
        Creates and runs the OpenSees model
        :param opt_sol: DataFrame                       Optimal solution (design solution)
        :param hinge_models: DataFrame                  Hinge models
        :param forces: DataFrame                        Acting forces
        :param say: float                               Yield spectral acceleration
        :param omega: float                             Overstrength factor
        :param t_upper: float                           Upper period limit
        :return: float, DataFrame                       Corrected overstrength factor and optimal solution
        """
        """Validate model first mode period
        Create an OpenSees model, run modal analysis and static pushover analysis"""
        model_periods, modalShape, gamma, mstar = self.ipbsd.ma_analysis(opt_sol, hinge_models, forces, self.fstiff)

        """Perform SPO analysis and calibrate the SPO shape"""
        spoResults = self.ipbsd.spo_opensees(opt_sol, hinge_models, forces, self.fstiff, modalShape)
        print("[SUCCESS] Modal analysis and  Static pushover analysis were successfully performed.")

        # Get the idealized version of the SPO curve and create a warningSPO = True if the assumed shape was incorrect
        d, v = self.derive_spo_shape(spoResults)

        # Calculate the actual yield strength and verify the overstrength ratio
        say_actual = v[1] / (9.81 * mstar * gamma)
        if say_actual / say > 1.1:
            # Modify omega
            omegaNew = omega / (say_actual / say)
            self.omegaWarn = True
        else:
            omegaNew = omega
            self.omegaWarn = False

        # There is a high likelihood that Fundamental period and SPO curve shape will not match the assumptions
        # Therefore the first iteration should correct both assumptions (further corrections are likely not to be large)
        if model_periods[0] > t_upper + model_periods[0] * 0.05:
            # Look for a new period
            tnew = t_upper - (model_periods[0] - opt_sol["T"])
            # Select all solutions in the vicinity of the new period
            sols = self.sols[(self.sols["T"] >= tnew - tnew * 0.05) & (self.sols["T"] <= tnew + tnew * 0.05)]
            # Select the solution with the least weight
            opt_sol = sols[sols["Weight"] == sols["Weight"].min()].iloc[0]
        else:
            tnew = model_periods[0]

        # Get new SPO parameters
        spo_data_updated = self.spo2ida_parameters(d, v, tnew)

        # Check whether the parameters vary from the original assumption (if True, then tolerance met)
        # NOTE: spo shape might loop around two scenarios (need a better technique to look for a solution)
        # It is primarily due to some low values (e.g. muC), where the large variations of e.g. a do not have a
        # significant impact on the final results, however are hard to fit
        self.spo_validate = all(list(map(self.compare_value, self.spo_data.values(), spo_data_updated.values())))
        self.spo_data = self.spo_data if self.spo_validate else spo_data_updated

        return omegaNew, opt_sol, gamma, mstar

    def validations(self, opt_sol, modes, sa, period_range, table_sls, t_lower, t_upper,
                    iterate=True, maxiter=20, omega=None):
        """
        Runs the iterations
        :param ipbsd: object                            IPBSD object
        :param opt_sol: DataFrame                       Optimal solution (design solution)
        :param modes: dict                              Modes for RMSA from optimal solution
        :param sa: ndarray                              Spectral accelerations at SLS
        :param period_range: ndarray                    Periods at SLS
        :param table_sls: dict                          Outputs of DBD at SLS
        :param t_lower: float                           Lower period
        :param t_upper: float                           Upper period
        :param iterate: bool                            Whether iterations to seek for better solutions are necessary
        :param maxiter: int                             Maximum number of iterations before the framework halts
        :param omega: float                             Overstrength ratio
        :return:
        """
        # Based on available literature, depending on perimeter or space frame, inclusion of gravity loads
        if omega is None:
            if self.ipbsd.data.n_gravity > 0:
                if self.analysis_type == 3 or self.analysis_type == 5:
                    omega = 1.3
                else:
                    omega = 1.0
            else:
                omega = 2.5

        print("[PHASE] Commencing phase 3...")
        part_factor, m_star, say, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omega)

        """Get action and demands"""
        print("[PHASE] Commencing phase 4...")
        phase_4 = self.iterate_phase_4(say, dy, sa, period_range, opt_sol, modes, table_sls, t_lower, t_upper)
        forces = next(phase_4)
        demands = next(phase_4)
        details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
        warn, warnings = next(phase_4)

        """Initialize iterations for corrections of assumptions and optimizations"""
        '''Sequence of corrections: 
        1. Period - correction of fstiff assumption on element basis; 
        2. Detailing - correction of c-s dimensions based on unsatisfactory detailing of EC8
        3. SPO shape - correction because of modification of Period, Detailing and/or nonchalant initial assumption
        '''
        omegaNew, opt_sol, part_factor, m_star = self.run_model(opt_sol, hinge_models, forces, say, omega, t_upper)

        # Perform iterations if required and if they were any warnings
        if iterate and (warn or not self.spo_validate or self.omegaWarn):
            """
            :param iterate: bool                    If iterations are needed
            :param warn: bool                       In detailing warnings of structural elements
            :param spo_validate: bool               SPO curve shape and fundamental period
            """
            # Warn about Overstrength ratio being large
            if self.omegaWarn and (not warn or self.spo_validate):
                print(f"[WARNING] Overstrength factor of {omegaNew} was larger than assumed value of {omega}. However, "
                      f"due to it being the only warning, no further iterations will be performed. It is advised to "
                      f"the user to manually alter the value of Overstrength and rerun analysis.")

            # Start with the correction of SPO curve shape. Also, recalculate the mode shape of new design solution
            cnt = 0
            print("[ITERATION 4] Commencing iteration...")
            while (warn or not self.spo_validate) and cnt+1 <= maxiter:

                """Look for a different solution"""
                # Get the new design solution and the modal shapes
                opt_sol, modes = self.seek_solution(warnings, opt_sol)

                # Print out the issue causing subsequent iteration to be run
                textToPrintOut = f"[ITERATION 4] Iteration: {cnt+1}. Rerunning phase 3 and/or 4 due to: "
                if self.omegaWarn:
                    textToPrintOut += "modified Overstrength"
                if warn:
                    textToPrintOut += ", unsatisfactory detailing" if self.omegaWarn else "unsatisfactory detailing"
                if not self.spo_validate:
                    if self.omegaWarn or warn:
                        textToPrintOut += ", SPO shape modification"
                    else:
                        textToPrintOut += "SPO shape modification"
                print(textToPrintOut)

                # Optimize for MAFC and find the new yield spectral acceleration
                part_factor, m_star, say, dy, spo2ida_data = self.iterate_phase_3(opt_sol, omegaNew, read=False)

                # Calculate demands and do detailing of structural elements
                phase_4 = self.iterate_phase_4(say, dy, sa, period_range, opt_sol, modes, table_sls, t_lower, t_upper)
                forces = next(phase_4)
                demands = next(phase_4)
                details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
                # Record any warnings regarding structural detailing
                warn, warnings = next(phase_4)

                # Run modal analysis and static pushover analysis and validate the assumptions
                omegaNew, opt_sol, part_factor, m_star = self.run_model(opt_sol, hinge_models, forces, say, omegaNew,
                                                                        t_upper)

                # Increase count of iterations
                cnt += 1

            if cnt == maxiter:
                print("[WARNING] Maximum number of iterations reached!")

        ipbsd_outputs = {"part_factor": part_factor, "Mstar": m_star, "Period range": [t_lower, t_upper],
                         "overstrength": omega, "yield": [say, dy], "lateral loads": forces}

        return ipbsd_outputs, spo2ida_data, opt_sol, demands, details, hinge_models, forces
