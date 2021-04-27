import copy
import numpy as np
from scipy.constants import h
from scipy.interpolate import PchipInterpolator

from superconductivity.fermi_functions import fermi
from superconductivity.multilayer.stack import Stack


def complex_conductivity(stacks, frequencies, temperatures=None, update=True,
                         squeeze=True):
    """
    Compute the complex conductivity (normalized to the normal state
    conductivity) at each z grid point in a stack.

    Args:
        stacks: iterable of superconductivity.multilayer.Stack
            Stack classes defining the geometry of the multilayer. There
            should be one for each temperature. If a single stack is
            given, the temperatures keyword argument can be used to
            broadcast the geometry to more than one temperature. If a
            stack is given outside of an iterable or the broadcasting is
            used, the stack is copied first.
        frequencies: iterable of floats
            The frequencies in Hz at which to evaluate the conductivity.
            All inputs are immediately coerced into numpy arrays.
        temperatures: iterable of floats
            If only one stack is provided, this keyword argument sets
            the temperatures at which to evaluate the conductivity. It
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
        sigma: numpy.ndarray
            The complex conductivity of the geometry. The axes
            correspond to (temperature, frequency, position). If any
            dimension has size 1 and the keyword argument 'squeeze' is
            True, it is removed.
    """
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
            stack.e = BCS / np.pi * stack.scale * np.concatenate(
                [np.linspace(0.0, 4.0, 4000),
                 np.logspace(np.log10(4.0), np.log10(32.0), 8001)[1:]])
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

    # Extract the energy vector adding two points at either end to make
    # sure the PCHIP interpolation extrapolates to a constant above max_e.
    max_e = stacks[0].e.max() * 10
    e = np.hstack([-2 * max_e, -max_e, -stacks[0].e[:0:-1], stacks[0].e,
                   max_e, 2 * max_e])

    # Create a fermi function that returns the right shape and also is
    # constant above max_e to maintain the constant PCHIP extrapolation.
    def f(en, temp):
        logic = (np.abs(en) >= max_e)
        if logic.any():
            energy = en.copy()
            energy[logic] = np.sign(en[logic]) * max_e
        else:
            energy = en
        return fermi(energy, temp, units='J')[:, np.newaxis]

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
            # data at all energies, appending the large energy limit to
            # each array.
            ones = np.ones((2, layer.theta.shape[1]))
            zeros = np.zeros_like(ones)
            dos_r = np.concatenate([ones,
                                    np.cos(layer.theta[:0:-1, :]).real,
                                    np.cos(layer.theta).real,
                                    ones], axis=0)
            dos_i = np.concatenate([zeros,
                                    -np.cos(layer.theta[:0:-1, :]).imag,
                                    np.cos(layer.theta).imag,
                                    zeros], axis=0)
            dop_r = np.concatenate([zeros,
                                    (1j * np.sin(layer.theta[:0:-1, :])).real,
                                    (-1j * np.sin(layer.theta)).real,
                                    zeros], axis=0)
            dop_i = np.concatenate([zeros,
                                    (-1j * np.sin(layer.theta[:0:-1, :])).imag,
                                    (-1j * np.sin(layer.theta)).imag,
                                    zeros], axis=0)

            # Create PCHIP functions from the data arrays.
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
                integrand1 = PchipInterpolator(e, integrand1,
                                               extrapolate=True, axis=0)
                integrand2 = PchipInterpolator(e, integrand2,
                                               extrapolate=True, axis=0)

                # Compute the complex conductivity at this frequency and
                # temperature for this layer in the stack. We choose an
                # arbitrarily large value for infinity since all of the
                # interesting information in the integrand should be close
                # to the gap energy.
                inf = max_e
                sigma1 = integrand1.integrate(-inf, inf)
                sigma2 = integrand2.integrate(-inf, inf)

                # Fill the output array.
                sigma[it, iv, z_start:z_stop] = sigma1 - 1j * sigma2

    # Remove extra dimensions
    if squeeze:
        sigma = sigma.squeeze()

    return sigma
