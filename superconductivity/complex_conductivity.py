import warnings
import numpy as np
import scipy.special as sp
import scipy.constants as sc
import scipy.integrate as it
import multiprocessing as mp

from superconductivity.fermi_functions import fermi
from superconductivity.density_of_pairs import dop_bcs, dop_dynes
from superconductivity.density_of_states import dos_bcs, dos_dynes
from superconductivity.gap_functions import delta_bcs, delta_dynes
from superconductivity.utils import coerce_arrays, BCS, combine_sigma


def value(temp, freq, tc, gamma=0, d=0, bcs=BCS, low_energy=False, parallel=False):
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
    bcs: bool (optional)
        Use the bcs constant where applicable. Only used for numeric
        computations of the complex conductivity. The default is
        superconductivity.utils.BCS.
    low_energy: bool (optional)
        Use the low energy limit formulas for the complex conductivity
        (h f << ∆ and kB T << ∆). It can dramatically speed up computation
        time. The default is False.
    parallel: multiprocessing.Pool or boolean (optional)
        A multiprocessing pool object to use for the computation. The default
        is False, and the computation is done in serial. If True, a Pool object
        is created with multiprocessing.cpu_count() CPUs. Only used if
        low_energy is False.
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
        sigma = numeric(temp, freq, tc, gamma=gamma, bcs=bcs, parallel=parallel)
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


def numeric(temp, freq, tc, gamma=0, bcs=BCS, parallel=False):
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
    parallel: multiprocessing.Pool or boolean (optional)
        A multiprocessing pool object to use for the computation. The default
        is False, and the computation is done in serial. If True, a Pool object
        is created with multiprocessing.cpu_count() CPUs.
    Returns
    -------
    sigma : numpy.ndarray, dtype=numpy.complex128
        The complex conductivity at temp and freq.
    """
    if gamma == 0:
        sigma = mattis_bardeen_numeric(temp, freq, tc, bcs=bcs, parallel=parallel)
    else:
        sigma = dynes_numeric(temp, freq, gamma, tc, bcs=bcs, parallel=parallel)
    return sigma


def mattis_bardeen_numeric(temp, freq, tc, bcs=BCS, parallel=False):
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
    parallel: multiprocessing.Pool or boolean (optional)
        A multiprocessing pool object to use for the computation. The default
        is False, and the computation is done in serial. If True, a Pool object
        is created with multiprocessing.cpu_count() CPUs.
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
    # make pool if needed
    close = True if parallel is True else False
    if parallel is True:
        parallel = mp.Pool(processes=mp.cpu_count())
    # compute the integral by looping over inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply inf by 0 warning
        if parallel:
            args = [(t[ii], w[ii]) for ii, _ in np.ndenumerate(temp)]
            sigma = parallel.starmap(mattis_bardeen_integral, args)
            sigma = np.array(sigma).reshape(temp.shape)
            if close:
                parallel.close()
                parallel.join()
        else:
            sigma = np.empty(temp.shape, dtype=np.complex)
            for ii, _ in np.ndenumerate(temp):
                sigma[ii] = mattis_bardeen_integral(t[ii], w[ii])
    return sigma


def dynes_numeric(temp, freq, gamma, tc, bcs=BCS, parallel=False):
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
    parallel: multiprocessing.Pool or boolean (optional)
        A multiprocessing pool object to use for the computation. The default
        is False, and the computation is done in serial. If True, a Pool object
        is created with multiprocessing.cpu_count() CPUs.
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
    g = np.full(temp.shape, gamma)
    # make pool if needed
    close = True if parallel is True else False
    if parallel is True:
        parallel = mp.Pool(processes=mp.cpu_count())
    # compute the integral by looping over inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # multiply inf by 0 warning
        if parallel:
            args = [(t[ii], w[ii], g[ii]) for ii, _ in np.ndenumerate(temp)]
            sigma = parallel.starmap(dynes_integral, args)
            sigma = np.array(sigma).reshape(temp.shape)
            if close:
                parallel.close()
                parallel.join()
        else:
            sigma = np.empty(temp.shape, dtype=np.complex)
            for ii, _ in np.ndenumerate(temp):
                sigma[ii] = dynes_integral(t[ii], w[ii], g[ii])
    return sigma


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
    k = 2 * (fermi(e, t) - fermi(e + w, t)) * (dos_bcs(e, 1, real=True) * dos_bcs(e + w, 1, real=True) +
                                               dop_bcs(e, 1, real=True) * dop_bcs(e + w, 1, real=True)) / w
    k[np.isnan(k)] = 0
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
    k = -(1 - 2 * fermi(e + w, t)) * (dos_bcs(e, 1, real=False) * dos_bcs(e + w, 1, real=True) +
                                      dop_bcs(e, 1, real=False) * dop_bcs(e + w, 1, real=True)) / w
    k[np.isnan(k)] = 0
    return k


def mattis_bardeen_integral(tii, wii):
    """
    Calculate the complex conductivity integral for a BCS superconductor at a
    single point.
    Parameters
    ----------
    tii: float
        reduced temperature (kB T / ∆)
    wii: float
        reduced frequency (h f / ∆)
    Returns
    -------
    sigma: numpy.ndarray
        The complex conductivity
    """
    sigma1 = it.quad(mattis_bardeen_kernel1, 1, np.inf, args=(tii, wii))[0]
    sigma2 = it.quad(mattis_bardeen_kernel2, 1 - wii, 1, args=(tii, wii))[0]
    return combine_sigma(sigma1, sigma2)


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
    k = (fermi(e, t) - fermi(e + w, t)) * (dos_dynes(e, 1, g, real=True) * dos_dynes(e + w, 1, g, real=True) +
                                           dop_dynes(e, 1, g, real=True) * dop_dynes(e + w, 1, g, real=True)) / w
    k[np.isnan(k)] = 0
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
    k = -(1 - 2 * fermi(e, t)) * (dos_dynes(e, 1, g, real=True) * dos_dynes(e + w, 1, g, real=False) +
                                  dop_dynes(e, 1, g, real=True) * dop_dynes(e + w, 1, g, real=False)) / w
    k[np.isnan(k)] = 0
    return k


def dynes_integral(tii, wii, gii):
    """
    Calculate the complex conductivity integral for a Dynes superconductor at a
    single point.
    Parameters
    ----------
    tii: float
        reduced temperature (kB T / ∆)
    wii: float
        reduced frequency (h f / ∆)
    gii: float
        reduced Dynes parameter (gamma / ∆).
    Returns
    -------
    sigma: numpy.ndarray
        The complex conductivity
    """
    points = (-1 - wii, -1, 1 - wii, 1)
    sigma1 = it.quad(dynes_kernel1, -1e2, 1e2, args=(tii, wii, gii), limit=200, points=points)[0]
    sigma2 = it.quad(dynes_kernel2, -1e2, 1e2, args=(tii, wii, gii), limit=200, points=points)[0]
    return combine_sigma(sigma1, sigma2)
