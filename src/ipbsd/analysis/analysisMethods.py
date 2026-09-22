import numpy as np
import pandas as pd

from analysis.openseesrun import OpenSeesRun


def run_simple_analysis(direction, solution, yield_sa, sls, data):
    if direction == 0:
        nbays = data.n_bays
    else:
        nbays = len(data.spans_y)

    print("[INITIATE] Starting simplified approximate demand estimation...")
    response = pd.DataFrame({'Mbi': np.zeros(data.nst),
                             'Mci': np.zeros(data.nst),
                             'Mce': np.zeros(data.nst)})

    # gravity acceleration, m/s2
    g = 9.81
    base_shear = yield_sa * solution["Mstar"] * solution["Part Factor"] * g
    masses = data.masses / data.n_seismic
    modes = [sls[str(st + 1)]['phi'] for st in range(data.nst)]
    # lateral forces
    forces = np.zeros(data.nst)
    # shear at each storey level
    shear = np.zeros(data.nst)
    for st in range(data.nst):
        forces[st] = masses[st] * modes[st] * base_shear / sum(map(lambda x, y: x * y, masses, modes))
    for st in range(data.nst):
        shear[st] = sum(fi for fi in forces[st:data.nst])

    # Demands on beams and columns in kNm
    # Assuming contraflexure point at 0.6h for the columns
    for st in range(data.nst):
        if st != data.nst - 1:
            response['Mbi'][st] = 1 / 2 / nbays * data.h[st] / 2 * (shear[st] + shear[st + 1])
        else:
            response['Mbi'][st] = 1 / 2 / nbays * data.h[st] / 2 * shear[st]
        # The following is based on assumption that beam stiffness effects are neglected
        ei_external = solution[f"he{st + 1}"] ** 4
        ei_internal = solution[f"hi{st + 1}"] ** 4
        ei_ratio = ei_internal / ei_external
        ei_total = 2 + ei_ratio * (nbays - 1)
        shear_external = shear[st] / ei_total
        shear_internal = shear_external * ei_ratio
        response['Mci'][st] = 0.6 * data.h[st] * shear_internal
        response['Mce'][st] = 0.6 * data.h[st] * shear_external


def run_opensees_analysis(direction, solution, hinge, data, action, fstiff, flag3d, pattern=None):
    """
    Runs OpenSees analysis
    :param direction: int
    :param solution: DataFrame
    :param hinge: dict
    :param data: dict
    :param action: list
    :param fstiff: float
    :param flag3d: bool
    :param pattern: ndarray                         Only for static pushover analysis (lateral analysis)
    :return:
    """

    if hinge is None:
        hinge = {"x_seismic": None, "y_seismic": None, "gravity": None}

    # call OpenSees object
    op = OpenSeesRun(data, solution, fstiff, hinge=hinge, direction=direction, system=data.configuration)

    # create the model
    if pattern is not None:
        op.create_model(gravity=True)
    else:
        op.create_model()

    # define masses
    op.define_masses()

    if not flag3d:
        # if 2D modelling, define pdelta-columns
        op.create_pdelta_columns(action)
        # number of modes to be considered
        num_modes = 1
    else:
        num_modes = data.nst if data.nst <= 9 else 9

    if pattern is not None:
        # Run static pushover (SPO) analysis
        return op.run_spo_analysis(load_pattern=2, mode_shape=pattern)
    else:
        # Run modal analysis
        return op.run_modal_analysis(num_modes)
