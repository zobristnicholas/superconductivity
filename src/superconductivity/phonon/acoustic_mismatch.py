import numpy as np
from numpy import arcsin, cos, sin, abs, nan, pi
from scipy.integrate import quad


def incident_l(angle, rho, ct, cl):
    """
    Compute the reflection and transmission probabilities for an incident
    longitudinal wave using the acoustic mismatch model from Kaplan et al.
    1979. (doi:10.1007/BF00119193)

    The material properties are ordered starting with the material containing
    the incident wave.

    The units for the material properties don't matter as long as they are the
    same for each material.

    Args:
        angle: float
            The incident angle in radians.
        rho: length 2 iterable of floats
            The densities of each material.
        ct: length 2 iterable of floats
            The transverse speed of sound in each material.
        cl: length 2 iterable of floats
            The longitudinal speed of sound in each material.

    Result:
        result: dictionary
            A dictionary containing the reflection, transmission
            probabilities, and resulting angles of each possible wave
            polarization.
    """
    if not 0 <= angle <= pi / 2:
        raise ValueError("Incident angles must be between 0 and pi / 2.")

    # Use Snell's law to calculate the transmitted and reflected angles
    theta1 = angle
    theta2 = arcsin(cl[1] * sin(theta1) / cl[0] + 0j)
    gamma1 = arcsin(ct[0] * sin(theta1) / cl[0] + 0j)
    gamma2 = arcsin(ct[1] * sin(theta1) / cl[0] + 0j)

    # Define some constants
    alpha1 = cos(theta1) / cl[0]  # w cancels out so we set it to 1
    alpha2 = cos(theta2) / cl[1]
    beta1 = cos(gamma1) / ct[0]
    beta2 = cos(gamma2) / ct[1]  # beta2 has the wrong sign in Kaplan eq A13
    sigma = sin(theta1) / cl[0]
    # sigma = sin(theta1) / cl[0] or sin(gamma1) / ct[0]
    #   or sin(theta2) / cl[1] or sin(gamma2) / ct[1]

    # Solve equations A14(a-d) in Kaplan 1979 (A = 1)
    lhs = np.array([
       [alpha1, sigma, alpha2, -sigma],
       [-sigma, beta1, sigma, beta2],
       [rho[0] * (cl[0]**2 * (alpha1**2 + sigma**2) - 2 * ct[0]**2 * sigma**2),
        2 * rho[0] * ct[0]**2 * beta1 * sigma,
        rho[1] * (2 * ct[1]**2 * sigma**2 - cl[1]**2 * (alpha2**2 + sigma**2)),
        2 * rho[1] * ct[1]**2 * beta2 * sigma],
       [2 * rho[0] * ct[0]**2 * alpha1 * sigma,
        rho[0] * ct[0]**2 * (sigma**2 - beta1**2),
        2 * rho[1] * ct[1]**2 * alpha2 * sigma,
        rho[1] * ct[1]**2 * (beta2**2 - sigma**2)]])
    rhs = np.array([
        alpha1,
        sigma,
        rho[0] * (2 * ct[0]**2 * sigma**2 - cl[0]**2 * (alpha1**2 + sigma**2)),
        2 * rho[0] * ct[0]**2 * alpha1 * sigma])
    x = np.linalg.solve(lhs, rhs)
    c, d, e, f = x  # transmission and reflection wave amplitudes

    # Enforce the critical angles
    with np.errstate(invalid='ignore'):  # number > NaN evaluates to False
        if theta1 > arcsin(cl[0] / ct[0]):
            d = 0  # reflected SV
        if theta1 > arcsin(cl[0] / cl[1]):
            e = 0  # transmitted L
        if theta1 > arcsin(cl[0] / ct[1]):
            f = 0  # transmitted SV

    # Use the energy conservation equation to get the transmission and
    # reflection coefficients. The denominator of the third term on the right
    # hand side of equation A16 in Kaplan 1979 has rho[1] where it should have
    # rho[0].
    r_l = (abs(c)**2).real
    r_sv = (abs(d)**2 * cl[0] * cos(gamma1) / (ct[0] * cos(theta1))).real
    t_l = (abs(e)**2 * (rho[1] * cl[0] * cos(theta2)
                        / (rho[0] * cl[1] * cos(theta1)))).real
    t_sv = (abs(f)**2 * (rho[1] * cl[0] * cos(gamma2)
                         / (rho[0] * ct[1] * cos(theta1)))).real

    result = dict(L=dict(T=dict(P=t_l, angle=theta2.real),
                         R=dict(P=r_l, angle=theta1.real)),
                  SV=dict(T=dict(P=t_sv, angle=gamma2.real),
                          R=dict(P=r_sv, angle=gamma1.real)),
                  SH=dict(T=dict(P=0., angle=nan), R=dict(P=0., angle=nan)))
    return result


def incident_sv(angle, rho, ct, cl):
    """
    Compute the reflection and transmission probabilities for an incident
    vertical shear wave using the acoustic mismatch model from Kaplan et al.
    1979. (doi:10.1007/BF00119193)

    The material properties are ordered starting with the material containing
    the incident wave.

    The units for the material properties don't matter as long as they are the
    same for each material.

    Args:
        angle: float
            The incident angle in radians.
        rho: length 2 iterable of floats
            The densities of each material.
        ct: length 2 iterable of floats
            The transverse speed of sound in each material.
        cl: length 2 iterable of floats
            The longitudinal speed of sound in each material.

    Result:
        result: dictionary
            A dictionary containing the reflection, transmission
            probabilities, and resulting angles of each possible wave
            polarization.
    """
    if not 0 <= angle <= pi / 2:
        raise ValueError("Incident angles must be between 0 and pi / 2.")

    # Use Snell's law to calculate the transmitted and reflected angles
    gamma1 = angle
    gamma2 = arcsin(ct[1] * sin(gamma1) / ct[0] + 0j)
    theta1 = arcsin(cl[0] * sin(gamma1) / ct[0] + 0j)
    theta2 = arcsin(cl[1] * sin(gamma1) / ct[0] + 0j)

    # Define some constants
    alpha1 = cos(theta1) / cl[0]  # w cancels out so we set it to 1
    alpha2 = cos(theta2) / cl[1]
    beta1 = cos(gamma1) / ct[0]
    beta2 = cos(gamma2) / ct[1]
    sigma = sin(theta1) / cl[0]
    # sigma = sin(theta1) / cl[0] or sin(gamma1) / ct[0]
    #   or sin(theta2) / cl[1] or sin(gamma2) / ct[1]

    # Solve the coupled boundary equations.
    lhs = np.array([
       [alpha1, sigma, alpha2, -sigma],
       [-sigma, beta1, sigma, beta2],
       [rho[0] * (cl[0]**2 * (alpha1**2 + sigma**2) - 2 * ct[0]**2 * sigma**2),
        2 * rho[0] * ct[0]**2 * beta1 * sigma,
        rho[1] * (2 * ct[1]**2 * sigma**2 - cl[1]**2 * (alpha2**2 + sigma**2)),
        2 * rho[1] * ct[1]**2 * beta2 * sigma],
       [2 * rho[0] * ct[0]**2 * alpha1 * sigma,
        rho[0] * ct[0]**2 * (sigma**2 - beta1**2),
        2 * rho[1] * ct[1]**2 * alpha2 * sigma,
        rho[1] * ct[1]**2 * (beta2**2 - sigma**2)]])
    rhs = np.array([-sigma,
                    beta1,
                    2 * rho[0] * ct[0]**2 * beta1 * sigma,
                    rho[0] * ct[0]**2 * (beta1**2 - sigma**2)])
    x = np.linalg.solve(lhs, rhs)
    c, d, e, f = x  # transmission and reflection wave amplitudes

    # Enforce the critical angles
    with np.errstate(invalid='ignore'):  # number > NaN evaluates to False
        if gamma1 > arcsin(ct[0] / cl[0]):
            c = 0  # reflected L
        if gamma1 > arcsin(ct[0] / cl[1]):
            e = 0  # transmitted L
        if gamma1 > arcsin(ct[0] / ct[1]):
            f = 0  # transmitted SV

    # Use the energy conservation equation to get the transmission and
    # reflection coefficients.
    r_l = (abs(c)**2 * ct[0] * cos(theta1) / (cl[0] * cos(gamma1))).real
    r_sv = (abs(d)**2).real
    t_l = (abs(e)**2 * (rho[1] * ct[0] * cos(theta2)
                        / (rho[0] * cl[1] * cos(gamma1)))).real
    t_sv = (abs(f)**2 * (rho[1] * ct[0] * cos(gamma2)
                         / (rho[0] * ct[1] * cos(gamma1)))).real

    result = dict(L=dict(T=dict(P=t_l, angle=theta2.real),
                         R=dict(P=r_l, angle=theta1.real)),
                  SV=dict(T=dict(P=t_sv, angle=gamma2.real),
                          R=dict(P=r_sv, angle=gamma1.real)),
                  SH=dict(T=dict(P=0., angle=nan), R=dict(P=0., angle=nan)))
    return result


def incident_sh(angle, rho, ct):
    """
    Compute the reflection and transmission probabilities for an incident
    horizontal shear wave using the acoustic mismatch model from Kaplan et al.
    1979. (doi:10.1007/BF00119193)

    The material properties are ordered starting with the material containing
    the incident wave.

    The units for the material properties don't matter as long as they are the
    same for each material.

    Args:
        angle: float
            The incident angle in radians.
        rho: length 2 iterable of floats
            The densities of each material.
        ct: length 2 iterable of floats
            The transverse speed of sound in each material.

    Result:
        result: dictionary
            A dictionary containing the reflection, transmission
            probabilities, and resulting angles of each possible wave
            polarization.
    """
    if not 0 <= angle <= pi / 2:
        raise ValueError("Incident angles must be between 0 and pi / 2.")

    # Use Snell's law to calculate the transmitted and reflected angles
    gamma1 = angle
    gamma2 = arcsin(ct[1] * sin(gamma1) / ct[0] + 0j)

    # Define some constants
    beta1 = cos(gamma1) / ct[0]  # w cancels out so we set it to 1
    beta2 = cos(gamma2) / ct[1]

    # Solve the boundary conditions.
    lhs = np.array([
        [-1, 1],
        [1, rho[1] * ct[1]**2 * beta2 / (rho[0] * ct[0]**2 * beta1)]
    ])
    rhs = np.array([1, 1])
    x = np.linalg.solve(lhs, rhs)
    d, f = x  # transmission and reflection wave amplitudes

    # Enforce the critical angle
    with np.errstate(invalid='ignore'):  # number > NaN evaluates to False
        if gamma1 > arcsin(ct[0] / ct[1]):
            f = 0  # transmitted SH

    # Use the energy conservation equation to get the transmission and
    # reflection coefficients.
    r_sh = (abs(d)**2).real
    t_sh = (abs(f)**2 * (rho[1] * ct[1] * cos(gamma2)
                         / (rho[0] * ct[0] * cos(gamma1)))).real

    result = dict(L=dict(T=dict(P=0., angle=nan), R=dict(P=0., angle=nan)),
                  SV=dict(T=dict(P=0., angle=nan), R=dict(P=0., angle=nan)),
                  SH=dict(T=dict(P=t_sh, angle=gamma2.real),
                          R=dict(P=r_sh, angle=gamma1.real)))
    return result


def transmission_l(rho, ct, cl):
    """
    Compute the angle-averaged transmission coefficient for longitudinal
    waves using the acoustic mismatch model from Kaplan et al. 1979.
    (doi:10.1007/BF00119193)

    The material properties are ordered starting with the material containing
    the incident wave.

    The units for the material properties don't matter as long as they are the
    same for each material.

    Args:
        rho: length 2 iterable of floats
            The densities of each material.
        ct: length 2 iterable of floats
            The transverse speed of sound in each material.
        cl: length 2 iterable of floats
            The longitudinal speed of sound in each material.

    Result:
        result: float
            The longitudinal transmission coefficient.
    """
    # Define the integrand for SH waves
    def integrand_l(theta1):
        long = incident_l(theta1, rho, ct, cl)
        eta = long["L"]["T"]["P"] + long["SV"]["T"]["P"]
        return 2 * eta * sin(theta1) * cos(theta1)

    # Compute the critical angles
    theta_c = []
    with np.errstate(invalid='ignore'):  # okay if theta_c = NaN
        theta_c.append(arcsin(cl[0] / ct[0]))  # reflected SV critical angle
        theta_c.append(arcsin(cl[0] / cl[1]))  # transmitted L critical angle
        theta_c.append(arcsin(cl[0] / ct[1]))  # transmitted SV critical angle
    theta_c = [angle for angle in theta_c if not np.isnan(angle)]
    if not theta_c:
        theta_c = None

    # Integrate and return the transmission coefficient for SV waves
    eta_l = quad(integrand_l, 0, pi / 2, points=theta_c)[0]

    return eta_l


def transmission_t(rho, ct, cl):
    """
    Compute the angle-averaged transmission coefficient for transverse
    waves using the acoustic mismatch model from Kaplan et al. 1979.
    (doi:10.1007/BF00119193)

    The material properties are ordered starting with the material containing
    the incident wave.

    The units for the material properties don't matter as long as they are the
    same for each material.

    Args:
        rho: length 2 iterable of floats
            The densities of each material.
        ct: length 2 iterable of floats
            The transverse speed of sound in each material.
        cl: length 2 iterable of floats
            The longitudinal speed of sound in each material.

    Result:
        result: float
            The transverse transmission coefficient.
    """
    # Define the integrand for SH waves
    def integrand_sh(gamma1):
        eta = incident_sh(gamma1, rho, ct)["SH"]["T"]["P"]
        return 2 * eta * sin(gamma1) * cos(gamma1)

    # Compute the critical angle
    gamma_c = [arcsin(ct[0] / ct[1])] if ct[0] <= ct[1] else None

    # Integrate and return the transmission coefficient for SH waves
    eta_sh = quad(integrand_sh, 0, pi / 2, points=gamma_c)[0]

    # Define the integrand for SV waves
    def integrand_sv(gamma1):
        sv = incident_sv(gamma1, rho, ct, cl)
        eta = sv["L"]["T"]["P"] + sv["SV"]["T"]["P"]
        return 2 * eta * sin(gamma1) * cos(gamma1)

    # Compute the critical angles
    gamma_c = []
    with np.errstate(invalid='ignore'):  # okay if gamma_c = NaN
        gamma_c.append(arcsin(ct[0] / cl[0]))  # reflected L critical angle
        gamma_c.append(arcsin(ct[0] / cl[1]))  # transmitted L critical angle
        gamma_c.append(arcsin(ct[0] / ct[1]))  # transmitted SV critical angle
    gamma_c = [angle for angle in gamma_c if not np.isnan(angle)]
    if not gamma_c:
        gamma_c = None

    # Integrate and return the transmission coefficient for SV waves
    eta_sv = quad(integrand_sv, 0, pi / 2, points=gamma_c)[0]

    return (eta_sh + eta_sv) / 2
