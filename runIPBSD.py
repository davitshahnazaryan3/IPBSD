from client.master import Master
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from external.getT1 import GetT1
from ipbsd import IPBSD
from pathlib import Path
from client.iterations import Iterations


path = Path.cwd()
outputPath = path.parents[0] / ".applications/LOSS Validation Manuscript/Case2"

# Add input arguments
analysis_type = 3
input_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Case2/ipbsd_input.csv"
hazard_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Hazard/Hazard-LAquila-Soil-C.pkl"
slfDir = outputPath / "slfoutput"
spo_file = path.parents[0] / ".applications/LOSS Validation Manuscript/Case2/spo.csv"
limit_eal = 1.0
mafc_target = 2.e-4
damping = .05
system = "Perimeter"
maxiter = 2
fstiff = 0.5
overstrength = 1.0
flag3d = True
export_cache = True
holdFlag = False
iterate = True
replCost = 349459.2 * 2

# Design solution to use (leave None, if the tool needs to look for the solution)
# solutionFile = path.parents[0] / ".applications/case1/designSol.csv"
solutionFileX = None
solutionFileY = None

method = IPBSD(input_file, hazard_file, slfDir, spo_file, limit_eal, mafc_target, outputPath, analysis_type,
               damping=damping, num_modes=2, iterate=iterate, system=system, maxiter=maxiter, fstiff=fstiff,
               flag3d=flag3d, solutionFileX=solutionFileX, solutionFileY=solutionFileY, export_cache=export_cache,
               holdFlag=holdFlag, overstrength=overstrength, rebar_cover=0.03, replCost=replCost)
start_time = method.get_init_time()

"""Calling the master file"""
ipbsd = Master(method.dir, method.flag3d)

"""Generating and storing the input arguments"""
ipbsd.read_input(method.input_file, method.hazard_file, method.outputPath)
# Initiate {project name}
print(f"[INITIATE] Starting IPBSD for {ipbsd.data.case_id}")

print("[PHASE] Commencing phase 1...")
method.create_folder(method.outputPath)
ipbsd.data.i_d["MAFC"] = method.target_MAFC
ipbsd.data.i_d["EAL"] = method.limit_EAL
# Store IPBSD inputs as a json
if method.export_cache:
    method.export_results(method.outputPath / "Cache/input_cache", ipbsd.data.i_d, "json")
print("[SUCCESS] Input arguments have been read and successfully stored")

# """Get EAL"""
lam_ls = ipbsd.get_hazard_pga(method.target_MAFC)
eal, y_fit, lam_fit = ipbsd.get_loss_curve(lam_ls, method.limit_EAL)
lossCurve = {"y": ipbsd.data.y, "lam": lam_ls, "y_fit": y_fit, "lam_fit": lam_fit, "eal": eal,
             "PLS": ipbsd.data.PLS}
print(f"[SUCCESS] EAL = {eal:.2f}%")

"""Get design limits"""
theta_max, a_max, slfsCache = ipbsd.get_design_values(method.slfDir, method.replCost)
if method.export_cache:
    method.export_results(method.outputPath / "Cache/SLFs", slfsCache, "pickle")
print("[SUCCESS] SLF successfully read, and design limits are calculated")

"""Transform design values into spectral coordinates"""
tables, delta_spectral, alpha_spectral = ipbsd.perform_transformations(theta_max, a_max)

if method.export_cache:
    method.export_results(method.outputPath / "Cache/table_sls", tables, "pickle")
print("[SUCCESS] Spectral values of design limits are obtained")

"""Get spectra at SLS"""
print("[PHASE] Commencing phase 2...")
sa, sd, period_range = ipbsd.get_spectra(lam_ls[1])
if method.export_cache:
    i = sa.shape[0]
    sls_spectrum = np.concatenate((period_range.reshape(i, 1), sd.reshape(i, 1), sa.reshape(i, 1)), axis=1)
    sls_spectrum = pd.DataFrame(data=sls_spectrum, columns=["Period", "Sd", "Sa"])
    method.export_results(method.outputPath / "Cache/sls_spectrum", sls_spectrum, "csv")

print("[SUCCESS] Response spectra generated")

"""Get feasible fundamental period range"""
if method.flag3d:
    period_limits = {"1": [], "2": []}
else:
    period_limits = {"1": []}

for i in range(delta_spectral.shape[0]):
    t_lower, t_upper = ipbsd.get_period_range(delta_spectral[i], alpha_spectral[i], sd, sa)
    period_limits[str(i+1)] = [t_lower, t_upper]
    print(f"[SUCCESS] Feasible period range identified. T_lower = {t_lower:.2f}s, T_upper = {t_upper:.2f}s")

# Check whether solutions file was provided
solution_x = method.check_file(method.solutionFileX)
solution_y = method.check_file(method.solutionFileY)

results = ipbsd.get_all_section_combinations(period_limits, fstiff=method.fstiff,
                                             cache_dir=method.outputPath / "Cache",
                                             solution_x=solution_x, solution_y=solution_y)
print("[SUCCESS] All section combinations were identified")

# Call the iterations function (iterations have not yet started though)
if method.flag3d:
    frames = 2
else:
    frames = 1

for i in range(frames):
    if i == 0:
        print("[INITIATE] Designing frame in X direction!")
    elif i == 1:
        print("[INITIATE] Designing frame in Y direction!")

    sols = results[i]["sols"]
    opt_sol = results[i]["opt_sol"]
    opt_modes = results[i]["opt_modes"]
    t_lower, t_upper = period_limits[str(i+1)]
    table_sls = tables[i]

    fin = Iterations(ipbsd, sols, method.spo_file, method.target_MAFC, method.analysis_type, method.damping,
                            method.num_modes, method.fstiff, method.rebar_cover, method.outputPath)

    # Run the validations and iterations if need be
    # ipbsd_outputs, spoResults, opt_sol, demands, details, hinge_models, action, modelOutputs = \
    #     fin.validations(opt_sol, opt_modes, sa, period_range, table_sls, t_lower, t_upper, method.iterate,
    #                            method.maxiter, omega=method.overstrength)
    omega = method.overstrength
    modes = opt_modes
    # Initialize period to use
    fin.period_to_use = None

    print("[PHASE] Commencing phase 3...")
    gamma, mstar, cy, dy, spo2ida_data = fin.iterate_phase_3(opt_sol, omega)

    """Get action and demands"""
    print("[PHASE] Commencing phase 4...")
    phase_4 = fin.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls)
    forces = next(phase_4)
    demands = next(phase_4)
    details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
    warnMax, warnMin, warnings = next(phase_4)
    # Reading SPO parameters from file
    read = True

    if iterate:
        # Start with the correction of SPO curve shape. Also, recalculate the mode shape of new design solution
        cnt = 0
        periodCache = opt_sol["T"]

        # Correction for period
        """Create an OpenSees model and run modal analysis"""
        model_periods, modalShape, gamma, mstar, opt_sol = fin.run_ma(opt_sol, hinge_models, forces, t_upper)
        # Update modes
        modes["Periods"] = np.array(model_periods)
        modes["Modes"][0, :] = np.array(modalShape)

        # Iterate until all conditions are met
        while (fin.warnT or warnMax or not fin.spo_validate or fin.omegaWarn) and cnt + 1 <= maxiter:
            # Iterations related to SPO corrections
            if not fin.spo_validate or fin.omegaWarn:
                # Reruns
                print("[RERUN] Rerun for SPO shape correction...")
                # Calculates the new cy for the corrected SPO shape, period, Overstrength and c-s
                gamma, mstar, cy, dy, spo2ida_data = fin.iterate_phase_3(opt_sol, omega, read=read)
                # Run elastic analysis and detail the structure
                phase_4 = fin.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls)
                forces = next(phase_4)
                demands = next(phase_4)
                details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
                warnMax, warnMin, warnings = next(phase_4)

                # Run modal analysis to check the T1
                model_periods, modalShape, gamma, mstar, opt_sol = fin.run_ma(opt_sol, hinge_models, forces,
                                                                              t_upper)

                # Update modes
                modes["Periods"] = np.array(model_periods)
                modes["Modes"][0, :] = np.array(modalShape)
                print("[RERUN COMPLETE] Rerun for SPO shape correction.")

            # Iterations related to Period corrections
            cntTrerun = 0
            while fin.warnT:
                if cntTrerun > 0:
                    rerun = True
                else:
                    rerun = False

                # Reruns
                print("[RERUN] Rerun for fundamental period correction...")
                gamma, mstar, cy, dy, spo2ida_data = fin.iterate_phase_3(opt_sol, omega, read=read)
                phase_4 = fin.iterate_phase_4(cy, dy, sa, period_range, opt_sol, modes, table_sls, rerun=rerun)
                forces = next(phase_4)
                demands = next(phase_4)
                details, hinge_models, hard_ductility, fract_ductility = next(phase_4)
                warnMax, warnMin, warnings = next(phase_4)

                # Run modal analysis
                model_periods, modalShape, gamma, mstar, opt_sol = fin.run_ma(opt_sol, hinge_models, forces, t_upper)

                # Update modes
                modes["Periods"] = np.array(model_periods)
                modes["Modes"][0, :] = np.array(modalShape)

                periodCache = opt_sol["T"]
                cntTrerun += 1
                if not fin.warnT:
                    # Update optimal solution (the update difference should be minimal as significant changes
                    # not expected)
                    opt_sol["Mstar"] = mstar
                    opt_sol["Part Factor"] = gamma
                    opt_sol["T"] = model_periods[0]
                    print("[RERUN COMPLETE] Rerun for fundamental period correction")

            # Exiting while warnT
            # Correction if unsatisfactory detailing, modifying only towards increasing c-s
            if warnMax:
                """Look for a different solution"""
                # Get the new design solution and the modal shapes
                opt_sol, modes = fin.seek_solution(warnings, opt_sol)
                print("[RERUN COMPLETE] New design solution has been selected due to unsatisfactory detailing...")

            """Create an OpenSees model and run static pushover analysis"""
            vy_assumed = cy * gamma * mstar * 9.81
            omegaCache = omega
            spoShapeCache = fin.spo_shape

            spoResults, spo_idealized, omega = fin.run_spo(opt_sol, hinge_models, forces, vy_assumed, modalShape,
                                                            omega)

            # Record OpenSees outputs
            fin.model_outputs = {"MA": {"T": model_periods, "modes": modalShape, "gamma": gamma, "mstar": mstar},
                                  "SPO": spoResults, "SPO_idealized": spo_idealized}

            # Reading SPO parameters from file
            read = False

            print("[SUCCESS] Static pushover analysis was successfully performed.")

            # Print out information
            print("--------------------------")
            print(f"[ITERATION {cnt + 1} END] Actual over assumed values of variables are provided: \n"
                  f"Yield strength overstrength: {omega / omegaCache * 100:.0f}%, \n"
                  f"Hardening ductility: {fin.spo_shape['mc'] / spoShapeCache['mc'] * 100:.0f}%, \n"
                  f"Fracturing ductility: {fin.spo_shape['mf'] / spoShapeCache['mf'] * 100:.0f}%, \n"
                  f"Fundamental period: {opt_sol['T'] / periodCache:.0f}%.")
            print("--------------------------")

            # Increase count of iterations
            cnt += 1

        if cnt == maxiter:
            print("[WARNING] Maximum number of iterations reached!")

    ipbsd_outputs = {"part_factor": gamma, "Mstar": mstar, "Period range": [t_lower, t_upper],
                     "overstrength": omega, "yield": [cy, dy], "muc": float(fin.spo_shape["mc"])}

# Export main outputs and cache
if method.export_cache:
    """Storing the outputs"""
    # Exporting the IPBSD outputs
    method.export_results(method.outputPath / "Cache/spoShape", pd.DataFrame(fin.spo_shape,
                                                                         index=[0]), "csv")
    method.export_results(method.outputPath / "Cache/lossCurve", lossCurve, "pickle")
    method.export_results(method.outputPath / "Cache/spoAnalysisCurveShape", spoResults, "pickle")
    method.export_results(method.outputPath / "optimal_solution", opt_sol, "csv")
    method.export_results(method.outputPath / "Cache/action", forces, "csv")
    method.export_results(method.outputPath / "Cache/demands", demands, "pkl")
    method.export_results(method.outputPath / "Cache/ipbsd", ipbsd_outputs, "pkl")
    method.export_results(method.outputPath / "Cache/details", details, "pkl")
    method.export_results(method.outputPath / "hinge_models", hinge_models, "csv")
    method.export_results(method.outputPath / "Cache/modelOutputs", fin.modelOutputs, "pickle")

    """Creating DataFrames to store for RCMRF input"""
    method.cacheRCMRF(ipbsd, details, opt_sol, demands)

method.get_time(start_time)
