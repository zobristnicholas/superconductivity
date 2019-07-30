import numpy as np
import numba as nb
import scipy.constants as sc

from superconductivity.utils import coerce_arrays


def fermi(en, temp, units='reduced'):
    """
    Calculate the Fermi Function given some energy and some temperature.
    Parameters
    ----------
    en : float, iterable of size N
        Energy relative to the fermi energy (E-Ef) in units of Joules or eV.
    temp : float, iterable of size N
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
    en, temp = coerce_arrays(en, temp)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    result = np.empty(en.size)
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
    _fermi(result, en, kbt)
    return result


@nb.njit
def _fermi(data, en, kbt):
    data[(kbt == 0) & (en < 0)] = 1
    data[(kbt == 0) & (en == 0)] = 0.5
    data[(kbt == 0) & (en > 0)] = 0
    data[kbt > 0] = 0.5 * (1 - np.tanh(0.5 * en[kbt > 0] / kbt[kbt > 0]))  # tanh() is better behaved than exp()

