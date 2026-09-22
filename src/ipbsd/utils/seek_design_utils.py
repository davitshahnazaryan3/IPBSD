import numpy as np

from utils.ipbsd_utils import getIndex, getEquation


def get_critical_designs(hinge_models_x, hinge_models_y):
    """
    Modify hinge elements of analysis seismic columns to the strongest (larger My) from designs of both directions
    Makes sure there is consistency between both directions of the building
    :param hinge_models_x: dict
    :param hinge_models_y: dict
    :return: dict
    """
    external_hinges_x = hinge_models_x[(hinge_models_x["Position"] == "analysis") &
                                       (hinge_models_x["Element"] == "Column")].reset_index()
    external_hinges_y = hinge_models_y[(hinge_models_y["Position"] == "analysis") &
                                       (hinge_models_y["Element"] == "Column")].reset_index()

    for index, row in external_hinges_x.iterrows():
        my_x = external_hinges_x["m1"].iloc[index]
        my_y = external_hinges_y["m1"].iloc[index]
        idx_x = external_hinges_x["index"].iloc[index]
        idx_y = external_hinges_y["index"].iloc[index]
        bay_n_x = external_hinges_x["Bay"].iloc[index]
        bay_n_y = external_hinges_y["Bay"].iloc[index]

        if my_x >= my_y:
            hinge_models_y.iloc[idx_y] = external_hinges_x.drop(columns=["index"]).iloc[index]
            # Modify corresponding Bay number
            hinge_models_y.at[idx_y, "Bay"] = bay_n_y

        else:
            hinge_models_x.iloc[idx_x] = external_hinges_y.drop(columns=["index"]).iloc[index]
            hinge_models_x.at[idx_x, "Bay"] = bay_n_x

    return hinge_models_x, hinge_models_y


def get_conservative_spo_shape(spo, residual=0.25):
    x = spo[0]
    y = spo[1]

    # All negatives to zero
    y[y < 0] = 0.0

    # Get maximum point for reference
    Vmax = max(y)

    # Get initial stiffness
    m1 = 0.2 * Vmax
    d1 = x[getIndex(m1, y)]
    stiff_elastic = m1 / d1

    # Get the yield point
    slopes = y / x
    stfIdx = np.where(slopes[1:] < 0.85 * stiff_elastic)[0][0]
    xint = x[stfIdx + 1]
    yint = y[stfIdx + 1]

    # Get the point of softening
    for i in range(len(x) - 1):
        stf = (y[i + 1] - y[i]) / (x[i] - x[i + 1])

        if stf > 50000:
            # High spikes
            ymax = y[i]
            xmax = x[i]
            break

    if "ymax" not in locals():
        ymax = max(y)
        xmax = x[getIndex(ymax, y)]

    # Make sure yield point is not larger than max point
    if yint > ymax:
        yint = ymax

    # # Residual point
    # yres = max(y[-1], residual * yint)
    # xres = x[i + getIndex(-yres, -y[i + 1:])]
    #
    # for i in range(len(x) - 1, 0, -1):
    #     if y[i] <= 0.0:
    #         y[i] = residual * yint
    #     if y[i - 1] / y[i] > 1.2 and y[i-1] > residual * yint:
    #         xres = x[i - 1]
    #         yres = y[i - 1]
    #         break
    # try:
    #     if yres > 0.35 * yint:
    #         yres = 0.35 * yint
    #         xres = x[i + getIndex(-yres, -y[i + 1:])]
    # except:
    #     pass

    # # Now, identify the residual strength point (here defined at V=0)
    # yres = max(y[-1], yint * residual)
    # idx = getIndex(1.01 * yres, y[::-1])
    # xres = x[::-1][idx]
    # # Getting the actual residual strength and corresponding displacement
    # ymin = yres
    #
    # # Select the softening slope until residual displacement
    # # Fitting based on the area under the softening slope
    # y_soft = y[getIndex(Vmax, y): getIndex(xres, x)]
    # nbins = len(y_soft) - 1
    # dx = (xres - xmax) / nbins
    # area_soft = np.trapz(y_soft, dx=dx)
    # xmin = 2 * area_soft / (Vmax + ymin) + xmax
    #
    # xres = xmin
    # yres = ymin

    # Using the Stiffness up till 0.8*ymax
    y_80 = 0.6 * yint
    idx = getIndex(1.01 * y_80, y[::-1])
    x_80 = x[::-1][idx]

    yres = residual * yint
    xres = (ymax - yres) * (x_80 - xmax) / (ymax - y_80) + xmax

    # Define the curve
    d = np.array([0., xint, xmax, xres])
    v = np.array([0., yint, ymax, yres])

    return d, v


def derive_spo_shape(spo, residual=0.1):
    """
    Fits a curve to the model SPO shape
    :param spo: dict                            Top displacement and base shear
    :param residual: float                      Percentage of Yield strength for residual
    :return: ndarrays                           Fitted top displacement and base shear
    """
    # Top displacement and base shear
    try:
        x = spo["d"]
        y = spo["v"]
    except:
        x = spo[0]
        y = spo[1]

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
    slope = m2 / d2

    # Fitting the plasticity portion based on the area under the curve
    y_pl = y[stfIdx: getIndex(Vmax, y)]
    nbins = len(y_pl) - 1
    dx = (dmax - d2) / nbins
    area_pl = np.trapz(y_pl, dx=dx)

    a = slope
    b = Vmax - slope * dmax
    c = 2 * area_pl - Vmax * dmax
    d = b ** 2 - 4 * a * c
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    if sol1 > 0 and sol2 > 0:
        xint = min(sol1, sol2)
    elif sol1 > 0 and sol2 <= 0:
        xint = sol1
    else:
        xint = sol2

    yint = xint * slope

    # Determinant is negative, look for an alternative fitting approach
    # print("[WARNING SPO FITTING] Using an approximate method of fitting, as a solution is not feasible!")
    # f = lambda x: (0.5 * (Vmax + x[0]) * (dmax - x[0] / x[1]) - area_pl)
    # x0 = [m2, slope]
    # sol = optimize.least_squares(f, x0)
    # yint = min(sol.x[0], 0.85 * Vmax)
    # xint = yint / sol.x[1]

    # if d < 0:
    #     # Determinant is negative, look for an alternative fitting approach
    #     print("[WARNING SPO FITTING] Using an approximate method of fitting, as a solution is not feasible!")
    #     f = lambda x: (0.5 * (Vmax + x[0]) * (dmax - x[0] / x[1]) - area_pl)
    #     x0 = [m2, slope]
    #     sol = optimize.least_squares(f, x0)
    #     yint = min(sol.x[0], 0.85 * Vmax)
    #     xint = yint / sol.x[1]
    # else:
    #     # Force Vy not be larger than maximum V
    #     yint = min(yint, 0.99 * Vmax)

    # Find point of plasticity initiation
    stf0 = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, len(x)):
        stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        if stf1 <= 0.85 * stf0:
            break
        else:
            stf0 = stf1

    if i == getIndex(Vmax, y):
        i = i - 10

    dPl = x[i]
    mPl = y[i]

    a0, b0 = getEquation((d1, m1), (d2, m2))
    a1, b1 = getEquation((dPl, mPl), (dmax, Vmax))

    # Find intersection point, i.e. the nominal yield point
    xint = (b1 - b0) / (a0 - a1)
    yint = a0 * xint + b0

    # Now, identify the residual strength point (here defined at V=0)
    yres = max(y[-1], yint * residual)
    idx = getIndex(1.01 * yres, y[::-1])
    xres = x[::-1][idx]
    # Getting the actual residual strength and corresponding displacement
    ymin = yres

    # ymin = y[-1]
    #
    # # Avoid negative residual strength and zero as residual strength
    # cnt = 2
    # while ymin <= 0:
    #     ymin = y[-cnt]
    #     cnt += 1

    # xmin = (Vmax - ymin) * (xres - dmax) / (Vmax - yres) + dmax

    # Select the softening slope until residual displacement
    # Fitting based on the area under the softening slope
    y_soft = y[getIndex(Vmax, y): getIndex(xres, x)]
    nbins = len(y_soft) - 1
    dx = (xres - dmax) / nbins
    area_soft = np.trapz(y_soft, dx=dx)
    xmin = 2 * area_soft / (Vmax + ymin) + dmax

    # Get the curve
    d = np.array([0., xint, dmax, xmin])
    v = np.array([0., yint, Vmax, ymin])

    return d, v


def derive_spo_shape_alternative(spo, residual=0.1):
    # Top displacement and base shear
    try:
        x = spo["d"]
        y = spo["v"]
    except:
        x = spo[0]
        y = spo[1]

    """
    The point below the max point is quite subjective
    So we need to identify two points
    Keep Vmax as the max point as long as the 
    Stiffness is reducing consistently
    And as long as the V is not varying from the max
    significantly.
    """
    # Identify the maximum point
    ymax = max(y)
    xmax = x[getIndex(ymax, y)]

    # Gradient of y
    grad = np.gradient(y)

    # Look for a very steep gradient
    idx = getIndex(10, -grad) - 1
    # Make sure that the new potential peak is not way lower than the maximum value
    # This new peak is due to P-delta effects
    if y[idx] / ymax >= 0.85:
        ymax = y[idx]
        xmax = x[idx]

    # Yield point
    # Get initial stiffness
    m1 = 0.2 * ymax
    d1 = x[getIndex(m1, y)]
    stiff_elastic = m1 / d1
    temp = y / x
    stfIdx = np.where(temp < 0.9 * stiff_elastic)[0][0]
    d2 = x[stfIdx]
    m2 = y[stfIdx]
    slope = m2 / d2

    # Fitting the plasticity portion based on the area under the curve
    y_pl = y[stfIdx: getIndex(ymax, y)]
    nbins = len(y_pl) - 1
    dx = (xmax - d2) / nbins
    area_pl = np.trapz(y_pl, dx=dx)

    a = slope
    b = ymax - slope * xmax
    c = 2 * area_pl - ymax * xmax
    d = b ** 2 - 4 * a * c
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    if sol1 > 0 and sol2 > 0:
        xint = min(sol1, sol2)
    elif sol1 > 0 and sol2 <= 0:
        xint = sol1
    else:
        xint = sol2

    yint = xint * slope

    # Find point of plasticity initiation
    stf0 = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, len(x)):
        stf1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        if stf1 <= 0.85 * stf0:
            break
        else:
            stf0 = stf1

    if i == getIndex(ymax, y):
        i = i - 10

    dPl = x[i]
    mPl = y[i]

    a0, b0 = getEquation((d1, m1), (d2, m2))
    a1, b1 = getEquation((dPl, mPl), (xmax, ymax))

    # Find intersection point, i.e. the nominal yield point
    xint = (b1 - b0) / (a0 - a1)
    yint = a0 * xint + b0

    # Now, identify the residual strength point (here defined at V=0)
    yres = max(y[-1], yint * residual)
    idx = getIndex(1.01 * yres, y[::-1])
    xres = x[::-1][idx]
    # Getting the actual residual strength and corresponding displacement
    ymin = y[-1]
    # xmin = (Vmax - ymin) * (xres - dmax) / (Vmax - yres) + dmax

    # Select the softening slope until residual displacement
    # Fitting based on the area under the softening slope
    y_soft = y[getIndex(ymax, y): getIndex(xres, x)]
    nbins = len(y_soft) - 1
    dx = (xres - xmax) / nbins
    area_soft = np.trapz(y_soft, dx=dx)
    xmin = 2 * area_soft / (ymax + ymin) + xmax

    # Avoid negative residual strength and zero as residual strength
    if ymin <= 0:
        ymin = 10.
    # Make sure that peak is not lower than yield point (incompatible for SPO2IDA)
    if yint > ymax:
        ymax = yint

    # Get the curve
    d = np.array([0., xint, xmax, xmin])
    v = np.array([0., yint, ymax, ymin])

    return d, v


def get_spo2ida_parameters(d, v, t):
    """
    Updates the SPO2IDA parameters
    :param d: ndarray                       Displacements
    :param v: ndarray                       Base shear forces
    :param t: float                         Fundamental period
    :return: dict                           Updated SPO2IDA input
    """
    # Residual strength
    r = v[-1] / v[1]
    # Hardening ductility
    muC = d[2] / d[1]
    # Fracturing ductility
    muF = d[3] / d[1]
    # Hardening slope
    a = (v[2] / v[1] - 1) / (muC - 1.)
    # Softening slope
    ap = (v[2] / v[1] - r) / (muC - muF)
    # Pinch weight
    pw = 1.0
    # Create a Dictionary
    spo_data = {"mc": muC, "a": a, "ac": ap, "r": r, "mf": muF, "pw": pw, "T": t}
    return spo_data


def find_solution(nst, period, solution, limit, tol, direction, **kwargs):
    period_1 = kwargs.get("period_1", None)
    limit_1 = kwargs.get("limit_1", None)

    for st in range(nst):
        # Internal columns of analysis frames along X
        solution[f"{direction}_seismic"][f"hi{st + 1}"] += 0.05
        # Gravity beams along X
        solution["gravity"][f"h{direction}{st + 1}"] += 0.05
        # Gravity columns
        if direction == "x":
            solution["gravity"][f"hi{st + 1}"] += 0.05
        else:
            if period_1 and period_1 <= tol * limit_1:
                solution["gravity"][f"hi{st + 1}"] += 0.05

        # If within XX% do not increase beam heights along corner frames
        if tol * limit / period < 0.75:
            solution[f"{direction}_seismic"][f"h{st + 1}"] += 0.05

        # If any of the cross-section dimensions is beyond an undesirable value, raise a warning
        vlist = [solution[f"{direction}_seismic"][f"hi{st + 1}"], solution[f"{direction}_seismic"][f"h{st + 1}"],
                 solution["gravity"][f"h{direction}{st + 1}"]]

        if not all(v < 0.95 for v in vlist):
            print("[WARNING] Cross-section dimensions are above 0.9m.")
    return solution
