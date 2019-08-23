import os
import warnings
import numpy as np
import scipy.special as sp
import scipy.constants as sc
import scipy.integrate as it
import scipy.optimize as opt
import scipy.interpolate as si

from superconductivity.utils import BCS
from superconductivity.fermi_functions import fermi
from superconductivity.density_of_pairs import dop_bcs, dop_dynes


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

_npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynes_gap.npz"))  # from dynes_numeric()
reduced_delta_dynes_interp = si.interp2d(_npz["t"], _npz["g"], _npz["dr"], kind="cubic")
reduced_delta_dynes_interp.__doc__ = """
    Return the reduced temperature dependent Dynes gap ∆(T)/∆(0) using
    an interpolation from data calculated from reduced_delta_dynes_numeric(). 
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    g: float
        The reduced Dynes parameter (gamma / delta0).
    Returns
    -------
    dr : float
        The reduced superconducting Dynes energy gap ∆(T)/∆(0).
    ----
    """


def reduced_delta_bcs(t, interp=True, approx=False):
    """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0).
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    interp: boolean (optional)
        Use interpolated values from reduced_delta_bcs_interp instead of
        recalculating numerically (much faster). The default is True.
    approx: boolean (optional)
        Use the low temperature approximation below 0.2 Tc instead of
        recalculating numerically (much faster). The default is False.
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
            dr[end] = reduced_delta_bcs_interp(t[end])
            i = reduced_delta_bcs_interp(t[crossover])
        else:
            dr[end] = reduced_delta_bcs_numeric(t[end])
            i = reduced_delta_bcs_numeric(t[crossover])
        dr[crossover] = a * (0.3 - t[crossover]) / 0.1 + i * (t[crossover] - 0.2) / 0.1
    else:
        logic = (t >= 0.005) & (t < 1)
        if interp:
            dr[logic] = reduced_delta_bcs_interp(t[logic])
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


def _self_consistent_bcs(dr, t):
    if dr == 0:
        return -np.inf
    return np.log(dr) - it.quad(_integrand_bcs, 0, np.inf, args=(dr, t))[0]


def _integrand_bcs(x, dr, t):
    return -2 * dop_bcs(x, 1j, real=False) * fermi(np.sqrt(x**2 + 1), t / BCS / dr)


def delta_bcs(temp, tc, bcs=BCS, interp=True, approx=False):
    """
    Return the temperature dependent BCS gap ∆(T).
    Parameters
    ----------
    temp : float or numpy.ndarray
        The temperature in Kelvin.
    tc: float
        The superconducting transition temperature.
    bcs: float (optional)
        The BCS constant. It isn't exactly constant across superconductors.
    interp: boolean (optional)
        Use interpolated values from reduced_delta_bcs_interp instead of
        recalculating numerically (much faster). The default is True.
    approx: boolean (optional)
        Use the low temperature approximation below 0.2 Tc instead of
        recalculating numerically (much faster). The default is False.
    Returns
    -------
    delta : float
        The superconducting BCS energy gap ∆(T).
    Notes
    ----
    """
    dr = reduced_delta_bcs(temp / tc, interp=interp, approx=approx)
    return bcs * sc.k * tc * dr


def reduced_delta_dynes(t, g, interp=True):
    """
    Return the reduced temperature dependent Dynes gap ∆(T)/∆(0).
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    g: float
        The reduced Dynes parameter (gamma / delta0).
    interp: boolean (optional)
        Use interpolated values from reduced_delta_dynes_interp instead of
        recalculating numerically (much faster). The default is True.
    Returns
    -------
    dr : float
        The reduced superconducting Dynes energy gap ∆(T)/∆(0).
    ----
    """
    # coerce inputs into numpy array and set up output array
    t = np.atleast_1d(t)
    dr = np.empty(t.shape)
    # set the gap out of the valid range to zero
    dr[np.logical_or(t >= 1, t < 0)] = 0
    logic = (t >= 0) & (t < 1)
    if interp:
        dr[logic] = reduced_delta_dynes_interp(t[logic], g)
    else:
        dr[logic] = reduced_delta_dynes_numeric(t[logic], g)
    return dr


def reduced_delta_dynes_numeric(t, g):
    """
    Return the reduced temperature dependent gap ∆(T)/∆(0) of a Dynes
    superconductor calculated numerically.
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    g: float
        The reduced Dynes parameter (gamma / delta0).
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    ----
    """
    if g == 0:
        return reduced_delta_bcs_numeric(t)
    t = np.atleast_1d(t)
    dr = np.empty(t.shape)
    for index, ti in enumerate(t):
        if ti == 0:
            dr[index] = 1
        elif ti == 1:
            dr[index] = 0
        else:
            dr[index] = opt.brentq(_self_consistent_dynes, 1, 0, args=(ti, g))
    return dr


def _self_consistent_dynes(dr, t, g):
    tc = _tc_dynes(g)
    if dr == 0:
        return -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            lhs = np.log(dr) + 0.5 * np.log(dr**2 * (2 * g * (g - np.sqrt(g**2 + 1)) + 1) /
                                                    (2 * g * (g - np.sqrt(g**2 + dr**2)) + dr**2))
        except RuntimeWarning:  # invalid value in log .. use low dr expansion
            lhs = (0.5 * np.log(8 * g * (g - np.sqrt(g**2 + 1)) + 4) + np.log(g) + 0.25 * (dr / g)**2 -
                   3 / 32 * (dr / g)**4 + 5 / 96 * (dr / g)**6 - 35 / 1024 * (dr / g)**8)
        rhs = (it.quad(_integrand_dynes, 0, 0.01, args=(dr, t, g, tc))[0] +
               it.quad(_integrand_dynes, 0.01, 0.1, args=(dr, t, g, tc))[0] +
               it.quad(_integrand_dynes, 0.1, 1, args=(dr, t, g, tc))[0] +
               it.quad(_integrand_dynes, 1, np.inf, args=(dr, t, g, tc))[0])
    return lhs - rhs


def _integrand_dynes(x, dr, t, g, tc):
    return -2 * dop_dynes(x, 1, g / dr) * fermi(x, t / BCS / dr * tc * (g + np.sqrt(g**2 + 1)))


def _tc_dynes_equation(tc, g):
    if tc == 0:
        return -np.inf
    return sp.digamma(0.5 + BCS / (2 * np.pi * tc) * g / (g + np.sqrt(g**2 + 1))) - sp.digamma(0.5) + np.log(tc)


def _tc_dynes(g):
    """Tc_dynes / Tc_bcs"""
    return opt.brentq(_tc_dynes_equation, 0, 1, args=g, xtol=1e-20)


def delta_dynes(temp, tc, g, bcs=BCS, interp=True):
    """
    Return the temperature dependent Dynes gap ∆(T).
    Parameters
    ----------
    temp : float or numpy.ndarray
        The temperature in Kelvin.
    tc: float
        The superconducting transition temperature.
    g: float
        The reduced Dynes parameter (gamma / delta0).
    bcs: float (optional)
        The BCS constant. It isn't exactly constant across superconductors.
    interp: boolean (optional)
        Use interpolated values from reduced_delta_bcs_interp instead of
        recalculating numerically (much faster). The default is True.
    Returns
    -------
    delta : float
        The superconducting Dynes energy gap ∆(T).
    Notes
    ----
    """
    dr = reduced_delta_dynes(temp / tc, g, interp=interp)

    return bcs * sc.k * tc * dr / ((g + np.sqrt(g**2 + 1)) * _tc_dynes(g))
