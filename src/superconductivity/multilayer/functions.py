import copy
import logging
import numpy as np
from scipy.constants import h, mu_0, epsilon_0
from scipy.interpolate import PchipInterpolator

from superconductivity.utils import BCS
from superconductivity.fermi_functions import fermi
from superconductivity.multilayer.stack import Stack

Z0 = np.sqrt(mu_0 / epsilon_0)  # free space impedance

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def complex_conductivity(stacks, frequencies, temperatures=None,
                         norm=True, update=True, squeeze=True):
    """
    Compute the (normalized) complex conductivity at each z grid point
    in a stack.

    Args:
        stacks: iterable of superconductivity.multilayer.Stack
            Stack classes defining the geometry of the multilayer. There
            should be one for each temperature. If a single stack is
            given, the temperatures keyword argument can be used to
            broadcast the geometry to more than one temperature. If a
            stack is given outside of an iterable or the broadcasting is
            used, the stack is copied first. Otherwise, the stack is updated
            inplace.
        frequencies: iterable of floats
            The frequencies in Hz at which to evaluate the conductivity.
            All inputs are immediately coerced into numpy arrays.
        temperatures: iterable of floats
            If only one stack is provided, this keyword argument sets
            the temperatures at which to evaluate the conductivity. It
            defaults to None, and the temperatures are determined by the
            temperatures of the stacks.
        norm: boolean
            The default is True and the conductivity is reported
            normalized to the normal state conductivity. If False, the
            complex conductivity is reported in units of Ohms^-1 m^-1.
        update: boolean
            The default is True, and the density of states will be
            computed in this function. To skip this computation, set
            this keyword argument to False.
        squeeze: boolean
            The default is True and singleton dimensions in the output
            array are removed.

    Returns:
        sigma: numpy.ndarray
            The complex conductivity of the geometry. The axes
            correspond to (temperature, frequency, position). If any
            dimension has size 1 and the keyword argument 'squeeze' is
            True, it is removed.
    """
    log.info("Computing the complex conductivity.")
    # Coerce the inputs into the right form.
    if isinstance(stacks, Stack):
        stacks = [copy.deepcopy(stacks)]

    if len(stacks) == 1 and temperatures is None:
        temperatures = np.array([stacks[0].t[0]])
    elif len(stacks) == 1 and temperatures is not None:
        temperatures = np.asarray(temperatures).ravel()
        stacks = [copy.deepcopy(stacks[0])] * temperatures.size
        for i, stack in enumerate(stacks):
            stack.t = temperatures[i]
    elif temperatures is not None:
        raise ValueError("The 'temperatures' keyword argument can only be "
                         "used if only one stack is provided.")
    else:
        temperatures = np.array([stack.t[0] for stack in stacks])

    frequencies = np.asarray(frequencies).ravel()

    # Check that there are the right number of stacks
    if len(stacks) != temperatures.size:
        raise ValueError("'stacks' must be a single stack or a list of "
                         "stacks of the same length as 'temperatures'.")

    for i, stack in enumerate(stacks):
        # If we are updating the stack, also use a better energy grid.
        if update:
            hf = h * frequencies.max() / BCS * np.pi / stack.scale
            prefactor = BCS / np.pi * stack.scale
            stack.e = prefactor * np.linspace(0.0, 64.0 + hf, 20000)
        # Check that all stacks have the same energy grid.
        if (stack.e != stacks[0].e).any():
            raise ValueError("The given 'stacks' do not have consistent "
                             "energy grids.")
        # Check that all stacks have the same position grid.
        if (stack.z != stacks[0].z).any():
            raise ValueError("The given 'stacks' do not have consistent "
                             "position grids.")
        # Check that all stacks have the right temperature.
        if (stack.t != temperatures[i]).any():
            raise ValueError("The given 'stacks' do not have consistent "
                             "temperatures.")

    # Create the output array.
    sigma = np.empty((temperatures.size, frequencies.size, stacks[0].z.size),
                     dtype=complex)

    # Get the maximum energy and frequency for the calculation
    max_e = stacks[0].e.max()
    max_hf = h * frequencies.max()

    # Extract the energy vector making it extend to negative energies.
    e = np.hstack([-stacks[0].e[:0:-1], stacks[0].e])

    # Create a fermi function that returns the right shape.
    def f(en, temp):
        return fermi(en, temp, units='J')[:, np.newaxis]

    # Loop over the given temperatures.
    for it, t in enumerate(temperatures):
        # Compute the density of states at this temperature.
        if update:
            stacks[it].update_dos()

        # Loop over the layers.
        z_stop = 0
        for layer in stacks[it].layers:
            z_start = z_stop
            z_stop = z_start + layer.z.size
            # Get the complex density of states and density of pairs
            # data at all energies.
            dos_r = np.concatenate([np.cos(layer.theta[:0:-1, :]).real,
                                    np.cos(layer.theta).real], axis=0)
            dos_i = np.concatenate([-np.cos(layer.theta[:0:-1, :]).imag,
                                    np.cos(layer.theta).imag], axis=0)
            dop_r = np.concatenate([(1j * np.sin(layer.theta[:0:-1, :])).real,
                                    (-1j * np.sin(layer.theta)).real], axis=0)
            dop_i = np.concatenate([(-1j * np.sin(layer.theta[:0:-1, :])).imag,
                                    (-1j * np.sin(layer.theta)).imag], axis=0)

            # Create PCHIP functions from the data arrays.
            with np.errstate(invalid="ignore"):
                dos_r = PchipInterpolator(e, dos_r, extrapolate=True, axis=0)
                dos_i = PchipInterpolator(e, dos_i, extrapolate=True, axis=0)
                dop_r = PchipInterpolator(e, dop_r, extrapolate=True, axis=0)
                dop_i = PchipInterpolator(e, dop_i, extrapolate=True, axis=0)

            # Loop over the given frequencies.
            for iv, v in enumerate(h * frequencies):
                # Create the integrand arrays.
                # Herman et al. Phys. Rev. B, 96, 1, 2017.
                integrand1 = (f(e, t) - f(e + v, t)) * (
                    dos_r(e) * dos_r(e + v) + dop_r(e) * dop_r(e + v)) / v
                integrand2 = -(1 - 2 * f(e, t)) * (
                    dos_r(e) * dos_i(e + v) + dop_r(e) * dop_i(e + v)) / v

                # Turn them into functions by PCHIP interpolation.
                with np.errstate(invalid="ignore"):
                    integrand1 = PchipInterpolator(e, integrand1,
                                                   extrapolate=True, axis=0)
                    integrand2 = PchipInterpolator(e, integrand2,
                                                   extrapolate=True, axis=0)

                # Compute the complex conductivity at this frequency and
                # temperature for this layer in the stack. We choose the
                # largest valid value for infinity. All of the interesting
                # information in the integrand should be close to the gap
                # energy anyways.
                pos_inf = max_e - max_hf
                neg_inf = -max_e - max_hf
                sigma1 = integrand1.integrate(neg_inf, pos_inf)
                sigma2 = integrand2.integrate(neg_inf, pos_inf)

                # Fill the output array.
                sigma[it, iv, z_start:z_stop] = sigma1 - 1j * sigma2

                # Give sigma units if we aren't normalizing it.
                if not norm:
                    sigma[it, iv, z_start:z_stop] /= layer.rho

    # Remove the extra dimensions.
    if squeeze:
        sigma = sigma.squeeze()

    return sigma


def surface_impedance(stacks, frequencies, temperatures=None, update=True,
                      squeeze=True):
    """
    Compute the surface impedance [Ohms] at the bottom and top of the stack.

    Args:
        stacks: iterable of superconductivity.multilayer.Stack
            Stack classes defining the geometry of the multilayer. There
            should be one for each temperature. If a single stack is
            given, the temperatures keyword argument can be used to
            broadcast the geometry to more than one temperature. If a
            stack is given outside of an iterable or the broadcasting is
            used, the stack is copied first. Otherwise, the stack is updated
            inplace.
        frequencies: iterable of floats
            The frequencies in Hz at which to evaluate the surface impedance.
            All inputs are immediately coerced into numpy arrays.
        temperatures: iterable of floats
            If only one stack is provided, this keyword argument sets
            the temperatures at which to evaluate the surface impedance. It
            defaults to None, and the temperatures are determined by the
            temperatures of the stacks.
        update: boolean
            The default is True, and the density of states will be
            computed in this function. To skip this computation, set
            this keyword argument to False.
        squeeze: boolean
            The default is True and singleton dimensions in the output
            array are removed.

    Returns:
        zs_bottom: numpy.ndarray
            The surface impedance at the bottom of the stack. The axes
            correspond to (temperature, frequency, position). If any
            dimension has size 1 and the keyword argument 'squeeze' is
            True, it is removed.
        zs_top: numpy.ndarray
            The surface impedance at the top of the stack. The axes
            correspond to (temperature, frequency, position). If any
            dimension has size 1 and the keyword argument 'squeeze' is
            True, it is removed.

    """
    log.info("Computing the surface impedance.")
    # Compute the complex conductivity.
    sigma = complex_conductivity(stacks, frequencies, norm=False,
                                 temperatures=temperatures, update=update,
                                 squeeze=False)  # temp x freq x pos

    # Make sure the stacks argument is a list.
    if isinstance(stacks, Stack):
        stacks = [stacks]

    # The complex conductivity has a discontinuity at layer boundaries. We
    # need to replace the double valued function by its average at each
    # boundary.
    z_list = [stacks[0].layers[0].z]
    z_size = stacks[0].layers[0].z.size
    sigma_list = [sigma[:, :, :z_size]]
    for layer in stacks[0].layers[1:]:
        z_list.append(layer.z[1:])
        sigma_list.append(sigma[:, :, z_size: layer.z.size + z_size])
        z_size += layer.z.size
    z = np.hstack(z_list)

    # The double z positions have been removed. Now we average sigma at the
    # boundary and create a new array with the same position grid as z.
    avg_list = []
    for i, s in enumerate(sigma_list[1:]):
        avg_list.append((sigma_list[i][:, :, -1] + s[:, :, 0]) / 2)
    sigma = np.concatenate(
        [sigma_list[0][:, :, :-1]]
        + [item for tup in zip(avg_list, sigma_list[1:])
           for item in (tup[0][:, :, np.newaxis], tup[1][:, :, 1:-1])]
        + [sigma_list[-1][:, :, -1:]], axis=-1)

    # We need sigma at the midpoint of the position grid for the algorithm
    # so we average one more time.
    sigma = (sigma[:, :, :-1] + sigma[:, :, 1:]) / 2
    dz = np.diff(z)

    # The abcd matrices for the stack can now be computed.
    ones = np.ones(sigma.shape)
    f = np.broadcast_to(frequencies[np.newaxis, :, np.newaxis], sigma.shape)
    abcd = np.array([[ones, 2j * np.pi * f * mu_0 * dz],
                     [sigma * dz, ones]])
    abcd = np.moveaxis(abcd, [0, 1], [-2, -1])  # (temp x freq x pos x 2 x 2)

    # We multiply the abcd matrices together to get the surface impedance.
    zs_v = np.array([[Z0], [1]])
    for i in range(abcd.shape[2]):  # loop over positions
        zs_v = abcd[:, :, i, :, :] @ zs_v  # (temp x freq x 2 x 1)
    zs_top = zs_v[:, :, 0, 0] / zs_v[:, :, 1, 0]  # (temp x freq)

    # Multiplying in the opposite order gives the impedance on the other
    # surface.
    zs_v = np.array([[Z0], [1]])
    for i in reversed(range(abcd.shape[2])):  # loop over positions
        zs_v = abcd[:, :, i, :, :] @ zs_v  # (temp x freq x 2 x 1)
    zs_bottom = zs_v[:, :, 0, 0] / zs_v[:, :, 1, 0]  # (temp x freq)

    # Remove the extra dimensions.
    if squeeze:
        zs_top = zs_top.squeeze()
        zs_bottom = zs_bottom.squeeze()

    return zs_bottom, zs_top
