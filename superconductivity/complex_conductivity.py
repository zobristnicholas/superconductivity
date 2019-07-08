import warnings
import numpy as np
import scipy.special as sp
import scipy.constants as sc
import scipy.integrate as it

from superconductivity.fermi_functions import fermi
from superconductivity.density_of_pairs import dop_bcs, dop_dynes
from superconductivity.density_of_states import dos_bcs, dos_dynes
from superconductivity.gap_functions import delta_bcs, delta_dynes
from superconductivity.utils import coerce_arrays, BCS, combine_sigma


def value(temp, freq, tc, gamma=0, d=0, low_energy=False, bcs=BCS):
    """
    Calculate the complex conductivity to normal conductivity ratio.
    Parameters
    ----------
    temp: float, iterable of size N
        Temperature in units of Kelvin.
    freq: float, iterable of size N
        Frequency in units of Hz.
    tc: float
        The transition temperature in units of Kelvin.
    gamma: float (optional)
        reduced Dynes parameter (gamma / ∆). The default is 0. It can only be
        used if low_energy is False.
    d: float (optional)
        Ratio of the imaginary gap to the real gap energy at zero temperature.
        It can only be used if low_energy is True.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). It can dramatically speed up computation
        time. The default is False.
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    if low_energy:
        if gamma != 0:
            raise ValueError("'gamma' can not be used with 'low_energy'")
        sigma = limit(temp, freq, tc, d=d, bcs=bcs)
    else:
        if d != 0:
            raise ValueError("'d' can only be used with 'low_energy'")
        sigma = numeric(temp, freq, tc, gamma=gamma, bcs=bcs)
    return sigma


def limit(temp, freq, tc, d=0, bcs=BCS):
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
    tc: float
        The transition temperature in units of Kelvin.
    d: float (optional)
        Ratio of the imaginary gap to the real gap energy at zero temperature.
    bcs: float (optional)
        BCS constant that relates the gap to the transition temperature.
        ∆ = bcs * kB * Tc. The default is superconductivity.utils.BCS.
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
    delta1 = delta_bcs(0, tc, bcs=bcs)
    delta2 = d * delta1
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
    return combine_sigma(sigma1, sigma2)


def numeric(temp, freq, tc, gamma=0, bcs=BCS):
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
    tc: float
        The transition temperature in units of Kelvin.
    gamma: float (optional)
        reduced Dynes parameter (gamma / ∆). The default is 0.
    bcs: float (optional)
        BCS constant that relates the gap to the transition temperature.
        ∆ = bcs * kB * Tc. The default is superconductivity.utils.BCS.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    if gamma == 0:
        sigma = mattis_bardeen_numeric(temp, freq, tc, bcs=bcs)
    else:
        sigma = dynes_numeric(temp, freq, gamma, tc, bcs=bcs)
    return sigma


def mattis_bardeen_numeric(temp, freq, tc, bcs=BCS):
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
    tc: float
        The transition temperature in units of Kelvin.
    bcs: float (optional)
        BCS constant that relates the gap to the transition temperature.
        ∆ = bcs * kB * Tc. The default is superconductivity.utils.BCS.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    # coerce inputs into numpy array
    temp, freq = coerce_arrays(temp, freq)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get the temperature dependent gap
    delta = delta_bcs(temp, tc, bcs=bcs)
    # calculate unitless reduced temperature and frequency
    t = temp * sc.k / delta
    w = sc.h * freq / delta
    # set the temperature independent bounds for integrals
    a1, b1, b2 = 1, np.inf, 1
    # allocate memory for arrays
    sigma1 = np.zeros(temp.shape)
    sigma2 = np.zeros(temp.shape)
    # compute the integral by looping over inputs
    for ii, _ in np.ndenumerate(temp):
        a2 = 1 - w[ii]
        sigma1[ii] = it.quad(mattis_bardeen_kernel1, a1, b1, args=(t[ii], w[ii]))[0]
        sigma2[ii] = it.quad(mattis_bardeen_kernel2, a2, b2, args=(t[ii], w[ii]))[0]

    return combine_sigma(sigma1, sigma2)


def dynes_numeric(temp, freq, gamma, tc, bcs=BCS):
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
    tc: float
        The transition temperature in units of Kelvin.
    gamma: float
        reduced Dynes parameter (gamma / ∆).
    bcs: float (optional)
        BCS constant that relates the gamma=0 gap to the gamma=0 transition
        temperature. ∆00 = bcs * kB * Tc0. The default is
        superconductivity.utils.BCS.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    # coerce inputs into numpy array
    temp, freq = coerce_arrays(temp, freq)
    assert (temp >= 0).all(), "Temperature must be >= 0."
    # get the temperature dependent gap
    delta = delta_dynes(temp, tc, gamma, bcs=bcs)
    # calculate unitless reduced temperature and frequency
    t = temp * sc.k / delta
    w = sc.h * freq / delta
    g = np.full(delta.shape, gamma)
    # allocate memory for arrays
    sigma1 = np.zeros(temp.shape)
    sigma2 = np.zeros(temp.shape)
    # compute the integral by looping over inputs
    for ii, _ in np.ndenumerate(temp):
        points = (-1 - w[ii], -1, 1 - w[ii], 1)
        sigma1[ii] = it.quad(dynes_kernel1, -1e2, 1e2, args=(t[ii], w[ii], g[ii]), limit=200, points=points)[0]
        sigma2[ii] = it.quad(dynes_kernel2, -1e2, 1e2, args=(t[ii], w[ii], g[ii]), limit=200, points=points)[0]

    return combine_sigma(sigma1, sigma2)


def mattis_bardeen_kernel1(e, t, w):
    """
    Calculate the Mattis-Bardeen kernel of the integral for sigma1 of the
    complex conductivity where E = hf < 2 ∆ (tones with frequency, f, do not
    break Cooper pairs).
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for sigma1
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply by inf
        k = 2 * (fermi(e, t) - fermi(e + w, t)) * (dos_bcs(e, 1, real=True) * dos_bcs(e + w, 1, real=True) +
                                                   dop_bcs(e, 1, real=True) * dop_bcs(e + w, 1, real=True)) / w
    return k


def mattis_bardeen_kernel2(e, t, w):
    """
    Calculate the Mattis-Bardeen kernel of the integral for sigma2 of the
    complex conductivity where E = hf < 2 ∆ (tones with frequency, f, do not
    break Cooper pairs).
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for sigma2
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply by inf
        k = -(1 - 2 * fermi(e + w, t)) * (dos_bcs(e, 1, real=False) * dos_bcs(e + w, 1, real=True) +
                                          dop_bcs(e, 1, real=False) * dop_bcs(e + w, 1, real=True)) / w
    return k


def dynes_kernel1(e, t, w, g):
    """
    Calculate the Dynes kernel of the integral for sigma1 of the complex
    conductivity where E = hf < 2 ∆ (tones with frequency, f, do not break
    Cooper pairs). From Herman et al. Phys. Rev. B, 96, 1, 2017.
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    g: float
        reduced Dynes parameter (gamma / ∆).
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for sigma1
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply by inf
        k = (fermi(e, t) - fermi(e + w, t)) * (dos_dynes(e, 1, g, real=True) * dos_dynes(e + w, 1, g, real=True) +
                                               dop_dynes(e, 1, g, real=True) * dop_dynes(e + w, 1, g, real=True)) / w
    return k


def dynes_kernel2(e, t, w, g):
    """
    Calculate the Dynes kernel of the integral for sigma2 of the complex
    conductivity where E = hf < 2 ∆ (tones with frequency, f, do not break
    Cooper pairs). From Herman et al. Phys. Rev. B, 96, 1, 2017.
    Parameters
    ----------
    e: numpy.ndarray
        reduced energy (E / ∆)
    t: float
        reduced temperature (kB T / ∆)
    w: float
        reduced frequency (h f / ∆)
    g: float
        reduced Dynes parameter (gamma / ∆).
    Returns
    -------
    k: numpy.ndarray
        The kernel for the integral for sigma2
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply by inf
        k = -(1 - 2 * fermi(e, t)) * (dos_dynes(e, 1, g, real=True) * dos_dynes(e + w, 1, g, real=False) +
                                      dop_dynes(e, 1, g, real=True) * dop_dynes(e + w, 1, g, real=False)) / w
    return k
