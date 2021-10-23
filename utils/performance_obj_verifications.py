"""
utility functions for verifying performance objectives, or other design objectives within IPBSD
"""
from utils.ipbsd_utils import success_msg, error_msg


def check_eal(eal, eal_limit):
    """
    Verifies if calculated EAL meets the limiting eal
    :param eal: float                                   Actual EAL computed as the area below the loss curve
    :param eal_limit: float                             EAL limit as a performance objective
    :return: bool                                       Whether the limit condition is met or not
    """
    if eal <= eal_limit:
        return True
    else:
        return False


def check_period(t, t_lower, t_upper, tol=1e-3, pflag=True):
    """
    T should be within a tolerable range
    :param t:                           Secant-to-yield period
    :param t_lower:                     Lower period bound
    :param t_upper:                     Upper period bound
    :param tol: float                   Tolerance for accuracy
    :param pflag: bool                  Print flag
    :return: bool
    """
    if t_lower - tol <= t <= t_upper + tol:
        if pflag:
            success_msg(f"T1 of {t} is in a range of {round(t_lower, 2)} and {round(t_upper, 2)}")
        return True
    else:
        if pflag:
            error_msg(f"[WARNING] T1 of {t} is NOT in a range of {round(t_lower, 2)} and {round(t_upper, 2)}")
        return False


def verify_period_range(t_lower, t_upper):
    """
    Warns if lower period is higher than upper period
    :param t_lower: float                       Lower period value in s
    :param t_upper: float                       Upper period value in s
    :return: None
    """
    if t_lower >= t_upper:
        raise ValueError(f"[EXCEPTION] Lower period of {t_lower:.2f}s is higher than upper period {t_upper:.2f}s. "
                         f"No solution found!")
