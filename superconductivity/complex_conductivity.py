import numpy as np
import scipy.special as sp
import scipy.constants as sc


def limit(temp, freq, delta1, delta2=0.):
    """Calculates the approximate complex conductivity in the limit hf << ∆ and
    kB T << ∆ given some temperature, frequency and transition temperature.
    Parameters
    ----------
    temp : float, numpy.ndarray
        Temperature in units of Kelvin.
    freq : float, numpy.ndarray
        Frequency in units of Hz.
    delta1: float
        Superconducting gap energy at 0 Kelvin in units of Joules.
    delta2: float (optional, default: 0)
        Complex superconducting gap energy at 0 Kelvin (finite loss at zero
        temperature) in units of Joules.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq."""
    assert np.all(temp > 0), "Temperature must be >= 0."
    size_t = np.size(temp)
    size_f = np.size(f)
    assert size_t == size_f or size_t == 0 or size_f == 0, "Incompatible array sizes"

    sigma1 = np.zeros(max(size_t, size_f))
    sigma2 = np.zeros(max(size_t, size_f))

    xi = sc.h * freq / (2 * sc.k * temp)
    eta = delta1 / (sc.k * temp)

    sigma1[temp == 0] = np.pi * delta2 / (sc.h * freq) + 1j * np.pi * delta1 / (sc.h * freq)
    sigma2[temp == 0] = np.pi * delta1 / (sc.h * freq)
    sigma1[temp != 0] = (4 * delta1 / (sc.h * freq) * np.exp(-eta) * np.sinh(xi) * sp.k0(xi) +
                         np.pi * delta2 / (sc.h * freq) *
                         (1 + 2 * delta1 / (sc.k * temp) * np.exp(-eta) * np.exp(-xi) * sp.i0(xi)))
    sigma2[temp != 0] = np.pi * delta1 / (sc.h * freq) * (1 - np.sqrt(2 * np.pi / eta) * np.exp(-eta) -
                                                          2 * np.exp(-eta) * np.exp(-xi) * sp.i0(xi))
    return sigma1 + 1j * sigma2


def quadrature(temp, freq, delta1):
    """Calculates the approximate complex conductivity by integrating given
    some temperature, frequency and transition temperature.
    Parameters
    ----------
    temp : float, numpy.ndarray
        Temperature in units of Kelvin.
    freq : float, numpy.ndarray
        Frequency in units of Hz.
    delta1: float
        Superconducting gap energy at 0 Kelvin in units of Joules.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq."""
    raise NotImplementedError
