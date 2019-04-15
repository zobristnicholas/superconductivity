import numba
import numpy as np
import scipy.constants as sc


def fermi(en, temp, units='reduced'):
    """
    Calculate the Fermi Function given some energy and some temperature.
    Parameters
    ----------
    en : float
        Energy relative to the fermi energy (E-Ef) in units of Joules or eV.
    temp : float
        Temperature in units of Kelvin.
    units : string (optional)
        Select units of energy. Acceptable values are ``'joules' or 'eV' or
        'reduced'``. Default is ``'reduced'``. Reduced units means both values
        are unitless, so onus is on the user to ensure that ``en/temp`` gives
        the desired result.
    Returns
    -------
    result : float
        The Fermi Function at en and temp.
    """
    # coerce inputs into numpy array and set up output array
    en = np.atleast_1d(en)
    temp = np.atleast_1d(temp)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    if temp.size == 1 and en.size != 1:
        temp = np.ones(en.size) * temp
    elif en.size == 1 and temp.size != 1:
        en = np.ones(temp.size) * en
    elif en.size != temp.size:
        raise ValueError("Incompatible array sizes")
    result = np.zeros(en.size)

    # convert temperature to joules
    if units in ['Joules', 'joules', 'j', 'J']:
        kbt = sc.k * temp
    # or eV if desired
    elif units in ['eV', 'ev', 'e']:
        kbt = sc.k * temp / sc.e
    # or allow for unitless quantities
    elif units in ['reduced', 'r']:
        kbt = temp
    else:
        raise ValueError("Unknown units requested.")

    # compute the fermi function
    result[np.logical_and(temp == 0, en < 0)] = 1
    result[np.logical_and(temp == 0, en > 0)] = 0
    result[temp > 0] = 0.5 * (1 - np.tanh(0.5 * en[temp > 0] / kbt[temp > 0]))  # tanh() is better behaved than exp()

    return result
