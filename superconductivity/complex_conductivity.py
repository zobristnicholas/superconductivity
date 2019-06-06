import numba
import numpy as np
import scipy.special as sp
import scipy.constants as sc
import scipy.integrate as it

from superconductivity.fermi_functions import fermi
from superconductivity.utils import coerce_arrays, BCS
from superconductivity.density_of_states import dos_bcs
from superconductivity.gap_functions import reduced_delta_bcs


def value(temp, freq, delta0, low_energy=False, gamma=0, bcs=BCS):
    """
    Calculate the complex conductivity to normal conductivity ratio.
    Parameters
    ----------
    temp: float, iterable of size N
        Temperature in units of Kelvin.
    freq: float, iterable of size N
        Frequency in units of Hz.
    delta0: float, complex
        Superconducting gap energy at 0 Kelvin in units of Joules. An imaginary
        part signifies finite loss at zero temperature.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). Can dramatically speed up computation time.
        Default is False.
    gamma: float (optional)
        Dynes parameter. The default is 0. Can only use if low_energy is False.
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    if low_energy:
        if gamma != 0:
            raise ValueError("'gamma' can not be used with 'low_energy'")
        if bcs != BCS:
            raise ValueError("'bcs' can not be used with 'low_energy'")
        sigma = limit(temp, freq, delta0)
    else:
        sigma = numeric(temp, freq, delta0=delta0, gamma=gamma, bcs=bcs)
    return sigma


def limit(temp, freq, delta0):
    """
    Calculate the approximate complex conductivity to normal conductivity
    ratio in the limit hf << ∆ and kB T << ∆ given some temperature, frequency
    and transition temperature.
    Parameters
    ----------
    temp : float, iterable of size N
        Temperature in units of Kelvin.
    freq : float, iterable of size N
        Frequency in units of Hz.
    delta0: float, complex
        Superconducting gap energy at 0 Kelvin in units of Joules. An imaginary
        part signifies finite loss at zero temperature.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    Notes
    -----
    Extension of Mattis-Bardeen theory to a complex gap parameter covered in
        Noguchi T. et al. Physics Proc., 36, 2012.
        Noguchi T. et al. IEEE Trans. Appl. SuperCon., 28, 4, 2018.
    The real part of the gap is assumed to follow the BCS temperature
        dependence expanded at low temperatures. See equation 2.53 in
        Gao J. 2008. CalTech. PhD dissertation.
    No temperature dependence is assumed for the complex portion of the gap
        parameter.
    """
    # coerce inputs into numpy array
    temp, freq = coerce_arrays(temp, freq)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # break up gap into real and imaginary parts
    delta1 = np.real(delta0)
    delta2 = np.imag(delta0)
    # allocate memory for complex conductivity
    sigma1 = np.zeros(freq.size)
    sigma2 = np.zeros(freq.size)
    # separate out zero temperature
    zero = (temp == 0)
    not_zero = (temp != 0)
    freq0 = freq[zero]
    freq1 = freq[not_zero]
    temp1 = temp[not_zero]
    # define some parameters
    xi = sc.h * freq1 / (2 * sc.k * temp1)
    eta = delta1 / (sc.k * temp1)
    # calculate complex conductivity
    sigma1[zero] = np.pi * delta2 / (sc.h * freq0)
    sigma2[zero] = np.pi * delta1 / (sc.h * freq0)
    sigma1[not_zero] = (4 * delta1 / (sc.h * freq1) * np.exp(-eta) * np.sinh(xi) * sp.k0(xi) +
                        np.pi * delta2 / (sc.h * freq1) *
                        (1 + 2 * delta1 / (sc.k * temp1) * np.exp(-eta) * np.exp(-xi) * sp.i0(xi)))
    sigma2[not_zero] = np.pi * delta1 / (sc.h * freq1) * (1 - np.sqrt(2 * np.pi / eta) * np.exp(-eta) -
                                                          2 * np.exp(-eta) * np.exp(-xi) * sp.i0(xi))
    return sigma1 - 1j * sigma2


def numeric(temp, freq, delta0, gamma=0, bcs=BCS):
    """
    Numerically calculate the complex conductivity to normal conductivity
    ratio by integrating given some temperature, frequency and transition
    temperature, where hf < ∆ (tones with frequency, f, do not break Cooper
    pairs).
    Parameters
    ----------
    temp : float, iterable of size N
        Temperature in units of Kelvin.
    freq : float, iterable of size N
        Frequency in units of Hz.
    delta0: float
        Superconducting gap energy at 0 Kelvin in units of Joules.
    gamma: float
        reduced Dynes parameter (gamma / ∆). The default is 0.
    bcs: float (optional)
        BCS constant that relates the gap to the transition temperature.
        ∆ = bcs * kB * Tc. The default is superconductivity.utils.BCS
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    # coerce inputs into numpy array
    temp, freq = coerce_arrays(temp, freq)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get the temperature dependent gap
    delta = delta0 * reduced_delta_bcs(bcs * sc.k * temp / delta0, bcs=bcs)
    # calculate unitless reduced temperature and frequency
    t = temp * sc.k / delta
    w = sc.h * freq / delta
    g = gamma / delta
    # set the temperature independent bounds for integrals
    a1, b1, b2 = 1, np.inf, 1
    # allocate memory for arrays
    sigma1 = np.zeros(temp.shape)
    sigma2 = np.zeros(temp.shape)
    # compute the integral by looping over inputs
    for ii in np.ndenumerate(temp.size):
        a2 = 1 - w[ii]
        sigma1[ii] = it.quad(sigma1_kernel, a1, b1, args=(t[ii], w[ii], g))[0]
        sigma2[ii] = it.quad(sigma2_kernel, a2, b2, args=(t[ii], w[ii], g))[0]

    return sigma1 - 1j * sigma2


def sigma1_kernel(e, t, w, g=0):
    """
    Calculate the kernel of the integral for the real part of the complex
    conductivity where E = hf < ∆ (tones with frequency, f, do not break Cooper
    pairs).
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    g: float (optional)
        reduced Dynes parameter (gamma / ∆). The default is 0.
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for the real part of the complex conductivity
    """
    k = (2 * (fermi(e, t) - fermi(e + w, t)) * (e ** 2 + w * e + 1) *
         dos_bcs(e, 1, gamma=g, norm=True) * dos_bcs(e + w, 1, gamma=g, norm=True)) / w
    return k


def sigma2_kernel(e, t, w, g=0):
    """
    Calculate the kernel of the integral for the imaginary part of the complex
    conductivity where E = hf < ∆ (tones with frequency, f, do not break Cooper
    pairs) for arcsin(1 - w) < y < pi / 2. Using e = sin(y) substitution in the
    dimensionless integral.
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    g: float
        reduced Dynes parameter (gamma / ∆). The default is 0.
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for the imaginary part of the complex
        conductivity
    """
    k = ((1 - 2 * fermi(e + w, t)) * (e**2 + w * e + 1) *
         (1j * dos_bcs(e, 1, gamma=g, abs_real=False, norm=True)).real * dos_bcs(e + w, 1, gamma=g, norm=True)) / w
    return k
