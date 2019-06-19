import os
import numpy as np
import scipy.constants as sc
import scipy.integrate as it
import scipy.optimize as opt
import scipy.interpolate as si

from superconductivity.utils import BCS


# These data points run from t=0.18 to t=1
# in steps of 0.02 from Muhlschlegel (1959)
_dr = [1.0, 0.9999, 0.9997, 0.9994, 0.9989, 0.9982, 0.9971, 0.9957, 0.9938, 0.9915, 0.9885, 0.985, 0.9809, 0.976,
       0.9704, 0.9641, 0.9569, 0.9488, 0.9399, 0.9299, 0.919, 0.907, 0.8939, 0.8796, 0.864, 0.8471, 0.8288, 0.8089,
       0.7874, 0.764, 0.7386, 0.711, 0.681, 0.648, 0.6117, 0.5715, 0.5263, 0.4749, 0.4148, 0.3416, 0.2436, 0.0]
_t = np.linspace(0.18, 1, len(_dr))
reduced_delta_bcs_muhlschlegel = si.interp1d(_t, _dr, kind="cubic")
reduced_delta_bcs_muhlschlegel.__doc__ = """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0) using
    an interpolation from data in Muhlschlegel (1959).
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    ----
    """

_npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcs_gap.npz"))  # from bcs_numeric()
reduced_delta_bcs_interp = si.interp1d(_npz["t"], _npz["dr"], kind="cubic")
reduced_delta_bcs_interp.__doc__ = """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0) using
    an interpolation from data calculated from reduced_delta_bcs_numeric(). 
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    ----
    """


def reduced_delta_bcs(t, interp=True, approx=True):
    """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0).
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    interp: boolean (optional)
        Use interpolated values from reduced_delta_bcs_interp instead of
        recalculating numerically (much faster).
    approx: boolean (optional)
        Use the low temperature approximation below 0.2 Tc instead of
        recalculating numerically (much faster).
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    ----
    """
    # coerce inputs into numpy array and set up output array
    t = np.atleast_1d(t)
    dr = np.empty(t.shape)
    # set the gap out of the valid range to zero
    dr[np.logical_or(t >= 1, t < 0)] = 0
    # for low temperatures set to 1 because floats don't have any more precision
    dr[np.logical_and(t >= 0, t < 0.005)] = 1
    if approx:
        beginning = (t >= 0.005) & (t <= 0.2)
        dr[beginning] = np.exp(-np.sqrt(2 * np.pi * t[beginning] / BCS) * np.exp(-BCS / t[beginning]))

        crossover = (t > 0.2) & (t <= 0.3)
        a = np.exp(-np.sqrt(2 * np.pi * t[crossover] / BCS) * np.exp(-BCS / t[crossover]))
        end = (t > 0.3) & (t < 1)
        if interp:
            dr[end] = reduced_delta_interp(t[end])
            i = reduced_delta_interp(t[crossover])
        else:
            dr[end] = reduced_delta_bcs_numeric(t[end])
            i = reduced_delta_bcs_numeric(t[crossover])
        dr[crossover] = a * (0.3 - t[crossover]) / 0.1 + i * (t[crossover] - 0.2) / 0.1
    else:
        logic = (t >= 0.005) & (t < 1)
        if interp:
            dr[logic] = reduced_delta_interp(t[logic])
        else:
            dr[logic] = reduced_delta_bcs_numeric(t[logic])
    return dr


def reduced_delta_bcs_numeric(t):
    """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0) calculated
    numerically.
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    ----
    """
    t = np.atleast_1d(t)
    dr = np.empty(t.shape)
    for index, ti in enumerate(t):
        if ti == 0:
            dr[index] = 1
        elif ti == 1:
            dr[index] = 0
        else:
            dr[index] = opt.brentq(_self_consistent_bcs, 1, 0, args=ti)
    return dr


def _self_consistent_bcs(dr, t=0):
    if dr == 0:
        return -np.inf
    return np.log(dr) + 2 * it.quad(_integrand_bcs, 0, np.inf, args=(dr, t))[0]


def _integrand_bcs(x, dr, t):
    return 1 / np.sqrt(x**2 + 1) / (np.exp(BCS * dr / t * np.sqrt(x**2 + 1)) + 1)


def delta_bcs(temp, tc, bcs=BCS, interp=True, approx=True):
    """
    Return the temperature dependent BCS gap ∆(T).
    Parameters
    ----------
    temp : float or numpy.ndarray
        The temperature in Kelvin.
    tc: float
        The superconducting transition temperature.
    bcs: float (optional)
        The bcs constant. It isn't exactly constant across superconductors.
    interp: boolean (optional)
        Use interpolated values from reduced_delta_bcs_interp instead of
        recalculating numerically (much faster).
    approx: boolean (optional)
        Use the low temperature approximation below 0.2 Tc instead of
        recalculating numerically (much faster).
    Returns
    -------
    delta : float
        The superconducting BCS energy gap ∆(T).
    Notes
    ----
    Tabulated data from Muhlschlegel (1959).
    Low-temperature analytic formula from Gao (2008).
    """
    dr = reduced_delta_bcs(temp / tc, interp=interp, approx=approx)
    return bcs * sc.k * tc * dr
