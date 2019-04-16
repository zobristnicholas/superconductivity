import numpy as np
import scipy.constants as sc
import scipy.interpolate as si

from superconductivity.utils import BCS


# These data points run from t=0.18 to t=1
# in steps of 0.02 from Muhlschlegel (1959)
_d = [1.0, 0.9999, 0.9997, 0.9994, 0.9989, 0.9982, 0.9971, 0.9957, 0.9938, 0.9915, 0.9885, 0.985, 0.9809, 0.976,
      0.9704, 0.9641, 0.9569, 0.9488, 0.9399, 0.9299, 0.919, 0.907, 0.8939, 0.8796, 0.864, 0.8471,  0.8288, 0.8089,
      0.7874, 0.764, 0.7386, 0.711, 0.681, 0.648, 0.6117, 0.5715, 0.5263, 0.4749, 0.4148, 0.3416, 0.2436, 0.0]
_t = np.linspace(0.18, 1, len(_d))
reduced_delta_muhlschlegel = si.InterpolatedUnivariateSpline(_t, _d, k=3)


def reduced_delta_bcs(t, bcs=BCS):
    """
    Return the reduced temperature dependent BCS gap ∆(T)/∆(0).
    Parameters
    ----------
    t : float or numpy.ndarray
        The reduced temperature  (T / Tc).
    bcs: float (optional)
        The bcs constant. It isn't exactly constant across superconductors.
    Returns
    -------
    dr : float
        The reduced superconducting BCS energy gap ∆(T)/∆(0).
    Notes
    ----
    Values of bcs other than the default result in a discontinuity at
        temp / tc = .3 since the tabulated data from Muhlschlegel is only
        valid for bcs = 1.764.
    Tabulated data from Muhlschlegel (1959).
    Low-temperature analytic formula from Gao (2008).
    """
    # coerce inputs into numpy array and set up output array
    t = np.atleast_1d(t)
    dr = np.zeros(t.shape)
    # set the gap out of the valid range to zero
    dr[np.logical_or(t >= 1, t < 0)] = 0
    # for temperatures close to tc use the numeric values from Muhlschlegel (1959)
    logic = np.logical_and(t < 1, t >= 0.3)
    dr[logic] = reduced_delta_muhlschlegel(t[logic])
    # This expression does a nice job of smoothly connecting the table to zero temp
    # Taken from Gao 2008
    logic = np.logical_and(t < 0.3, t >= 0.005)
    dr[logic] = np.exp(-np.sqrt(2 * np.pi * t[logic] / bcs) * np.exp(-bcs / t[logic]))
    # for low temperatures set to 1 because floats don't have any more precision
    dr[np.logical_and(t >= 0, t < 0.005)] = 1

    return dr


def delta_bcs(temp, tc, bcs=BCS):
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
    Returns
    -------
    delta : float
        The superconducting BCS energy gap ∆(T).
    Notes
    ----
    Values of bcs other than the default result in a discontinuity at
        temp / tc = .3 since the tabulated data from Muhlschlegel is only
        valid for bcs = 1.764.
    Tabulated data from Muhlschlegel (1959).
    Low-temperature analytic formula from Gao (2008).
    """
    dr = reduced_delta_bcs(temp / tc, bcs=bcs)
    return bcs * sc.k * temp * dr
