import numba
import numpy as np
import scipy.constants as sc


@numba.jit("float64(float64, float64, string)")
def fermi(en, temp, units='reduced'):
    """Calculate the Fermi Function given some energy and some temperature.
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
        The Fermi Function at en and temp."""
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
    # skip calculation if something easy is requested
    if en == 0:
        result = 0.5
    elif temp == 0:
        if en < 0:
            result = 1
        elif en > 0:
            result = 0
        else:
            raise ValueError("Energy must be a real number.")
    # compute Fermi function (equivalent to 1 / (1 + e^(en/kbt))
    # tanh is better behaved for very large or small values of en/kbt
    elif temp > 0:
        result = 0.5 * (1 - np.tanh(0.5 * en / kbt))
    else:
        raise ValueError("Temperature must be >= 0.")
    return result
