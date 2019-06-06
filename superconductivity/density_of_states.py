import numpy as np
from numpy.lib.scimath import sqrt  # sqrt that doesn't error on negative floats


def dos_bcs(en, delta, gamma=0, amplitude=0, ep=0, fwhm=0, abs_real=True, norm=False):
    """
    Compute the density of states for a generic BCS superconductor with the
    option of a small Lorentzian distribution of inter-gap states. Functional
    form from Noguchi T. et al. IEEE Trans. Appl. SuperCon., 28, 4, 2018.
    Parameters
    ----------
    en: float, numpy.ndarray
        Energy relative to the fermi energy (E-Ef) in any units.
    delta: float, complex
        Superconducting gap energy in units of en. An imaginary part signifies
        finite loss at zero temperature.
    gamma: float (optional)
        Dynes parameter for broadening the density of states in units of en.
        Typically not used if delta is complex.
        Dynes et al. Phys. Rev. Lett., 41, 21, 1978.
    amplitude: float (optional)
        Area under the Lorentzian curve for the inter-gap states. If zero
        (default) ep and fwhm are ignored.
    ep: float (optional)
        Energy relative to the fermi energy in units of en for the Lorentzian
        distribution.
    fwhm: float (optional)
        Full-width-half-max energy in units of en for the Lorentzian
        distribution. Cannot be zero if amplitude is not zero.
    abs_real: boolean (optional)
        If False, the absolute value of the real part of the result is not
        taken. This would not return the true density of states but is useful
        for computing integrals where -delta < en < delta. The default is True.
    norm: boolean (optional)
        If True, normalize the density of states by the energy. This is useful
        if multiplying by a function that contains 1 / energy to avoid
        singularities. If en = 0, nothing is done. The default is False.
    Returns
    -------
    dos: numpy.ndarray
        density of states as a function of en
    """
    en = np.atleast_1d(en)
    dos = np.empty(en.shape, dtype=np.complex)
    # determine density of states
    if norm:
        logic = (en != 0)
        dos[logic] = (en[logic] - 1j * gamma) / sqrt((en[logic] - 1j * gamma)**2 - delta**2) / en[logic]
        logic = np.logical_not(logic)
        dos[logic] = (-1j * gamma) / sqrt((-1j * gamma)**2 - delta**2)
    else:
        dos = (en - 1j * gamma) / sqrt((en - 1j * gamma)**2 - delta**2)
    # get real value or fix branch cut
    if abs_real:
        dos = np.abs(np.real(dos))
    else:
        # make sure that the branch cut on sqrt doesn't mess things up
        if norm:
            logic = (en < 0) & (dos.imag > 0)
            dos[logic] = np.conj(dos[logic])
        else:
            logic = (en < 0) & (dos.imag < 0)
            dos[logic] = np.conj(dos[logic])
        logic = (en > 0) & (dos.imag > 0)
        dos[logic] = np.conj(dos[logic])
    # add an optional lorentzian distribution
    if amplitude != 0:
        if fwhm != 0:
            if norm:
                logic = (en != 0)
                dos[logic] += 2 * amplitude * fwhm / np.pi / (4 * (en[logic] - ep) ** 2 + fwhm ** 2) / en[logic]
                logic = np.logical_not(logic)
                dos[logic] += 2 * amplitude * fwhm / np.pi / (4 * (en[logic] - ep) ** 2 + fwhm ** 2)
            else:
                dos += 2 * amplitude * fwhm / np.pi / (4 * (en - ep)**2 + fwhm**2)
        else:
            raise ValueError("If the amplitude is not zero, fwhm must not be zero")
    return dos
