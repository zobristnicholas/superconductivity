subroutine solve_diffusion_equation(energies, z, theta_old, order, boundaries, &
                                    interfaces, tol, d, rho, z_scale, &
                                    z_guess, n_layers, n_points, n_guess, &
                                    n_energy, n_min, theta)
    use bvp_m
    implicit none
    integer, parameter :: dp=kind(1d0)
    integer, intent(in) :: n_layers, n_points, n_guess, n_energy, n_min
    integer, intent(in) :: interfaces(n_layers + 1)
    real(kind=dp), intent(in) :: boundaries(n_layers - 1), order(n_points)
    real(kind=dp), intent(in) :: energies(n_energy), z(n_points),  tol
    real(kind=dp), intent(in) :: d(n_layers), rho(n_layers), z_scale(n_layers)
    real(kind=dp), intent(in) :: z_guess(n_guess)
    real(kind=dp), intent(in) :: theta_old(2 * n_layers, n_min)
    real(kind=dp), intent(out) :: theta(n_energy, n_points)

    real(kind=dp) :: zp(size(z)), dorder(size(z)), y(2 * n_layers, size(z))
    real(kind=dp) :: y_guess(2 * n_layers)
    integer :: n_eqns, start, stop, i, ii, err
    type(bvp_sol) :: sol
    real(kind=dp), save :: energy
    !$omp threadprivate(energy)

    ! There are two times as many equations as there are layers.
    n_eqns = 2 * n_layers

    ! Create a PCHIP interpolation of the order parameter for each layer.
    do i = 1, n_layers
        start = interfaces(i) + 1
        stop = interfaces(i + 1)
        call dpchez(stop - start + 1, z(start:stop), order(start:stop), &
                    dorder(start:stop), .false., 0, 0, err)
    end do

    !$omp parallel do private(y_guess, sol, start, stop, zp, y)
    do i = 1, n_energy
        energy = energies(i)  ! threadprivate global variable

        !Determine the best guess for y for this energy.
        y_guess = theta_old(:, min(i, n_min))

        ! Solve the diffusion equation.
        sol = bvp_init(n_eqns, n_layers, z_guess, y_guess)
        sol = bvp_solver(sol, f, bc, dfdy=jac, tol=tol, trace=0)

        ! Return the solution
        do ii = 1, n_layers
            start = interfaces(ii) + 1
            stop = interfaces(ii + 1)
            zp(start:stop) = transform(z(start:stop), ii)
            call bvp_eval(sol, zp(start:stop), y(:, start:stop))
            theta(i, start:stop) = y(2 * ii - 1, start: stop)
        end do

        ! Deallocate solution memory
        call bvp_terminate(sol)
    end do
    !$omp end parallel do

    contains
        function transform(z, ii) result(zp)
            implicit none
            real(kind=dp), intent(in) :: z(:)
            integer, intent(in) :: ii
            real(kind=dp) :: zp(size(z))

            if (mod(ii, 2) == 1) then
                zp = (z - sum(d(1:ii - 1))) / d(ii)
            else
                zp = 1 - (z - sum(d(1:ii - 1))) / d(ii)
            end if
        end function transform


        function itransform(zp, ii) result(z)
            implicit none
            real(kind=dp), intent(in) :: zp(:)
            integer, intent(in) :: ii
            real(kind=dp) :: z(size(zp))

            if (mod(ii, 2) == 1) then
                z = d(ii) * zp + sum(d(1:ii - 1))
            else
                z = d(ii) * (1 - zp) + sum(d(1:ii - 1))
            end if
        end function itransform


        function delta(zp, ii) result(r)
            implicit none
            real(kind=dp), intent(in) :: zp
            integer, intent(in) :: ii
            real(kind=dp) :: rs(1)
            real(kind=dp) :: r
            real(kind=dp) :: zpp(1)
            integer :: start, stop
            ! Undo transform to put zp in normal coordinates
            zpp = itransform([zp], ii)

            start = interfaces(ii) + 1
            stop = interfaces(ii + 1)
            ! Extract the interpolated data
            call dpchfe(stop - start + 1, z(start:stop), order(start:stop), &
                        dorder(start:stop), 1, .false., 1, zpp, rs, err)
            r = rs(1)
        end function delta


        subroutine f(z, y, dydz)
            implicit none
            real(kind=dp), intent(in) :: z, y(n_eqns)
            real(kind=dp), intent(out) :: dydz(n_eqns)
            integer :: ii

            do ii = 1, n_layers
                dydz(2 * ii - 1) = y(2 * ii)
                dydz(2 * ii) = (z_scale(ii))**2 &
                        * (energy * sin(y(2 * ii - 1)) &
                                - delta(z, ii) * cos(y(2 * ii - 1)))
            end do
        end subroutine f


        subroutine jac(z, y, dfdy)
            implicit none
            real(kind=dp), intent(in) :: z, y(n_eqns)
            real(kind=dp), intent(out) :: dfdy(n_eqns, n_eqns)
            integer :: ii

            dfdy(:, :) = 0.0_dp
            do ii = 1, n_layers
                dfdy(2 * ii - 1, 2 * ii) = 1.0_dp
                dfdy(2 * ii, 2 * ii - 1) = (z_scale(ii))**2 &
                        * (energy * cos(y(2 * ii - 1)) &
                                + delta(z, ii) * sin(y(2 * ii - 1)))
            end do
        end subroutine jac


        subroutine bc(ya, yb, bca, bcb)
            implicit none
            real(kind=dp), intent(in) :: ya(n_eqns), yb(n_eqns)
            real(kind=dp), intent(out) :: bca(n_layers), bcb(n_layers)

            real(kind=dp) :: ratio1, ratio2
            integer :: ii

            ! To separate the boundary conditions we apply transform() to z so
            ! a = 0 and b = 1. The new coordinates reverse the direction of the
            ! even layers so the sign of the derivative in the usual boundary
            ! conditions gets swapped.
            do ii = 1, n_layers
                ! Set the boundary condition for the first outer edge.
                if (ii == 1) then
                    bca(ii) = ya(2)
                end if

                ! Set the boundary conditions for odd layers.
                if (mod(ii, 2) == 1) then
                    ! The odd layers only have right boundary conditions.
                    if (ii == n_layers) then
                        bcb(ii) = yb(n_eqns)
                    else
                        ratio1 = rho(ii) * d(ii) / (rho(ii + 1) * d(ii + 1))
                        ratio2 = z_scale(ii) / boundaries(ii)
                        bcb(2 * ii - 1) = yb(2 * ii) + ratio1 * yb(2 * ii + 2)
                        bcb(2 * ii) = yb(2 * ii) &
                                - ratio2 * sin(yb(2 * ii + 1) - yb(2 * ii - 1))
                    end if
                ! Set the boundary conditions for even layers.
                else
                    ! The even layers only have left boundary conditions.
                    if (ii == n_layers) then
                        bca(ii) = ya(n_eqns)
                    else
                        ratio1 = rho(ii) * d(ii) / (rho(ii + 1) * d(ii + 1))
                        ratio2 = z_scale(ii) / boundaries(ii)
                        bca(2 * ii - 2) = ya(2 * ii) + ratio1 * ya(2 * ii + 2)
                        bca(2 * ii - 1) = ya(2 * ii) &
                                + ratio2 * sin(ya(2 * ii + 1) - ya(2 * ii - 1))
                    end if
                end if
            end do
        end subroutine bc
end subroutine solve_diffusion_equation
