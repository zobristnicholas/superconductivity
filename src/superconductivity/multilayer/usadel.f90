subroutine solve_imaginary(energies, z, theta_old, order, boundaries, &
                           interfaces, tol, n_threads, d, rho, z_scale, &
                           z_guess, n_layers, n_points, n_guess, n_energy, &
                           n_min, theta, info)
    use omp_lib
    use bvp_m
    implicit none
    integer, parameter :: dp=kind(1d0)
    integer, intent(in) :: n_layers, n_points, n_guess, n_energy, n_min
    integer, intent(in) :: interfaces(n_layers + 1), n_threads
    real(kind=dp), intent(in) :: boundaries(n_layers - 1), order(n_points)
    real(kind=dp), intent(in) :: energies(n_energy), z(n_points),  tol
    real(kind=dp), intent(in) :: d(n_layers), rho(n_layers), z_scale(n_layers)
    real(kind=dp), intent(in) :: z_guess(n_guess)
    real(kind=dp), intent(in) :: theta_old(2 * n_layers, n_min)
    real(kind=dp), intent(out) :: theta(n_energy, n_points)
    integer, intent(out) :: info

    real(kind=dp) :: zp(size(z)), dorder(size(z)), y(2 * n_layers, size(z))
    real(kind=dp) :: y_guess(2 * n_layers)
    integer :: n_eqns, start, stop, i, ii, err
    type(bvp_sol) :: sol
    real(kind=dp), save :: energy
    !$omp threadprivate(energy)

    ! Set the computation info to 0. Any other value indicates an error.
    info = 0

    ! Set the maximum number of threads to use for the calculation.
    call omp_set_num_threads(n_threads)

    ! There are two times as many equations as there are layers.
    n_eqns = 2 * n_layers

    ! Create a PCHIP interpolation of the order parameter for each layer.
    do i = 1, n_layers
        start = interfaces(i) + 1
        stop = interfaces(i + 1)
        call dpchez(stop - start + 1, z(start:stop), order(start:stop), &
                    dorder(start:stop), .false., 0, 0, err)
    end do

    ! Loop over the energies at which we want to solve the BVP.
    !$omp parallel do private(i, ii, y_guess, sol, start, stop, zp, y)
    do i = 1, n_energy
        ! Only do the loop if no loops have had problems.
        if (info == 0) then
            energy = energies(i)  ! threadprivate global variable

            !Determine the best guess for y for this energy.
            y_guess = theta_old(:, min(i, n_min))

            ! Solve the diffusion equation.
            sol = bvp_init(n_eqns, n_layers, z_guess, y_guess)
            sol = bvp_solver(sol, f, bc, dfdy=jac, tol=tol, trace=0, &
                             stop_on_fail=.false.)

            ! Check to see if there were problems.
            if (sol%info /= 0) then
                info = sol%info
            else
                ! Return the solution.
                do ii = 1, n_layers
                    start = interfaces(ii) + 1
                    stop = interfaces(ii + 1)
                    zp(start:stop) = transform(z(start:stop), ii)
                    call bvp_eval(sol, zp(start:stop), y(:, start:stop))
                    theta(i, start:stop) = y(2 * ii - 1, start:stop)
                end do

                ! Deallocate solution memory.
                call bvp_terminate(sol)
            end if
        end if
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
            ! Undo transform to put zp in normal coordinates.
            zpp = itransform([zp], ii)

            start = interfaces(ii) + 1
            stop = interfaces(ii + 1)
            ! Extract the interpolated data.
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
                dydz(2 * ii) = z_scale(ii) * (energy * sin(y(2 * ii - 1)) &
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
                dfdy(2 * ii, 2 * ii - 1) = z_scale(ii) &
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
                        ratio2 = boundaries(ii) / rho(ii) * d(ii)
                        bcb(2 * ii - 1) = yb(2 * ii) + ratio1 * yb(2 * ii + 2)
                        bcb(2 * ii) = ratio2 * yb(2 * ii) &
                                - sin(yb(2 * ii + 1) - yb(2 * ii - 1))
                    end if
                ! Set the boundary conditions for even layers.
                else
                    ! The even layers only have left boundary conditions.
                    if (ii == n_layers) then
                        bca(ii) = ya(n_eqns)
                    else
                        ratio1 = rho(ii) * d(ii) / (rho(ii + 1) * d(ii + 1))
                        ratio2 = boundaries(ii) / rho(ii) * d(ii)
                        bca(2 * ii - 2) = ya(2 * ii) + ratio1 * ya(2 * ii + 2)
                        bca(2 * ii - 1) = ratio2 * ya(2 * ii) &
                                + sin(ya(2 * ii + 1) - ya(2 * ii - 1))
                    end if
                end if
            end do
        end subroutine bc
end subroutine solve_imaginary


subroutine solve_real(energies, z, theta_old, order, boundaries, interfaces, &
                      tol, n_threads, d, rho, z_scale, z_guess, n_layers, &
                      n_points, n_guess, n_energy, n_min, theta, info)
    use omp_lib
    use bvp_m
    implicit none
    integer, parameter :: dp=kind(1d0)
    integer, intent(in) :: n_layers, n_points, n_guess, n_energy, n_min
    integer, intent(in) :: interfaces(n_layers + 1), n_threads
    real(kind=dp), intent(in) :: boundaries(n_layers - 1), order(n_points)
    real(kind=dp), intent(in) :: energies(n_energy), z(n_points),  tol
    real(kind=dp), intent(in) :: d(n_layers), rho(n_layers), z_scale(n_layers)
    real(kind=dp), intent(in) :: z_guess(n_guess)
    complex(kind=dp), intent(in) :: theta_old(2 * n_layers, n_min)
    complex(kind=dp), intent(out) :: theta(n_energy, n_points)
    integer, intent(out) :: info

    real(kind=dp) :: zp(size(z)), dorder(size(z)), y(4 * n_layers, size(z))
    real(kind=dp) :: y_guess(4 * n_layers)
    integer :: n_eqns, start, stop, i, ii, err
    type(bvp_sol) :: sol
    real(kind=dp), save :: energy
    complex(kind=dp), parameter :: j = (0, 1)
    !$omp threadprivate(energy)

    ! Set the computation info to 0. Any other value indicates an error.
    info = 0

    ! Set the maximum number of threads to use for the calculation.
    call omp_set_num_threads(n_threads)

    ! There are two times as many equations as there are layers.
    n_eqns = 2 * n_layers

    ! Create a PCHIP interpolation of the order parameter for each layer.
    do i = 1, n_layers
        start = interfaces(i) + 1
        stop = interfaces(i + 1)
        call dpchez(stop - start + 1, z(start:stop), order(start:stop), &
                    dorder(start:stop), .false., 0, 0, err)
    end do

    ! Loop over the energies at which we want to solve the BVP.
    !$omp parallel do private(i, ii, y_guess, sol, start, stop, zp, y)
    do i = 1, n_energy
        ! Only do the loop if no loops have had problems.
        if (info == 0) then
            energy = energies(i)  ! threadprivate global variable

            !Determine the best guess for y for this energy.
            y_guess(:n_eqns) = theta_old(:, min(i, n_min))%re
            y_guess(n_eqns + 1:) = theta_old(:, min(i, n_min))%im

            ! Solve the diffusion equation.
            sol = bvp_init(2 * n_eqns, 2 * n_layers, z_guess, y_guess)
            sol = bvp_solver(sol, f, bc, dfdy=jac, tol=tol, trace=0, &
                             stop_on_fail=.false.)

            ! Check to see if there were problems.
            if (sol%info /= 0) then
                info = sol%info
            else
                ! Return the solution.
                do ii = 1, n_layers
                    start = interfaces(ii) + 1
                    stop = interfaces(ii + 1)
                    zp(start:stop) = transform(z(start:stop), ii)
                    call bvp_eval(sol, zp(start:stop), y(:, start:stop))
                    theta(i, start:stop) = y(2 * ii - 1, start:stop) &
                        + j * y(n_eqns + 2 * ii - 1, start:stop)
                end do

                ! Deallocate solution memory.
                call bvp_terminate(sol)
            end if
        end if
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
            ! Undo transform to put zp in normal coordinates.
            zpp = itransform([zp], ii)

            start = interfaces(ii) + 1
            stop = interfaces(ii + 1)
            ! Extract the interpolated data.
            call dpchfe(stop - start + 1, z(start:stop), order(start:stop), &
                        dorder(start:stop), 1, .false., 1, zpp, rs, err)
            r = rs(1)
        end function delta


        subroutine f(z, y, dydz)
            implicit none
            real(kind=dp), intent(in) :: z, y(2 * n_eqns)
            real(kind=dp), intent(out) :: dydz(2 * n_eqns)
            integer :: ii
            complex(kind=dp) :: yc, dydzc

            do ii = 1, n_layers
                yc = y(2 * ii - 1) + j * y(n_eqns + 2 * ii - 1)
                dydzc = -z_scale(ii) &
                        * (j * energy * sin(yc) + delta(z, ii) * cos(yc))
                ! real part
                dydz(2 * ii - 1) = y(2 * ii)
                dydz(2 * ii) = dydzc%re
                ! imaginary part
                dydz(n_eqns + 2 * ii - 1) = y(n_eqns + 2 * ii)
                dydz(n_eqns + 2 * ii) = dydzc%im
            end do
        end subroutine f


        subroutine jac(z, y, dfdy)
            implicit none
            real(kind=dp), intent(in) :: z, y(2 * n_eqns)
            real(kind=dp), intent(out) :: dfdy(2 * n_eqns, 2 * n_eqns)
            integer :: ii
            complex(kind=dp) :: yc, dfdyc

            dfdy(:, :) = 0.0_dp
            do ii = 1, n_layers
                yc = y(2 * ii - 1) + j * y(n_eqns + 2 * ii - 1)
                dfdyc = z_scale(ii) &
                        * (-j * energy * cos(yc) + delta(z, ii) * sin(yc))
                ! real part
                dfdy(2 * ii - 1, 2 * ii) = 1.0_dp
                dfdy(2 * ii, 2 * ii - 1) = dfdyc%re
                ! imaginary part
                dfdy(n_eqns + 2 * ii - 1, n_eqns + 2 * ii) = 1.0_dp
                dfdy(n_eqns + 2 * ii, n_eqns + 2 * ii - 1) = dfdyc%re
                ! real / imaginary cross terms
                dfdy(2 * ii, n_eqns + 2 * ii - 1) = -dfdyc%im
                dfdy(n_eqns + 2 * ii, 2 * ii - 1) = dfdyc%im

            end do
        end subroutine jac


        subroutine bc(ya, yb, bca, bcb)
            implicit none
            real(kind=dp), intent(in) :: ya(2 * n_eqns), yb(2 * n_eqns)
            real(kind=dp), intent(out) :: bca(2 * n_layers), bcb(2 * n_layers)

            real(kind=dp) :: ratio1, ratio2
            integer :: ii
            complex(kind=dp) :: yc, bcc

            ! To separate the boundary conditions we apply transform() to z so
            ! a = 0 and b = 1. The new coordinates reverse the direction of the
            ! even layers so the sign of the derivative in the usual boundary
            ! conditions gets swapped.
            do ii = 1, n_layers
                ! Set the boundary condition for the first outer edge.
                if (ii == 1) then
                    bca(ii) = ya(2)  ! real
                    bca(n_layers + ii) = ya(n_eqns + 2)  ! imaginary
                end if

                ! Set the boundary conditions for odd layers.
                if (mod(ii, 2) == 1) then
                    ! The odd layers only have right boundary conditions.
                    if (ii == n_layers) then
                        bcb(ii) = yb(n_eqns)  ! real
                        bcb(n_layers + ii) = yb(n_eqns + n_eqns)  ! imaginary
                    else
                        ratio1 = rho(ii) * d(ii) / (rho(ii + 1) * d(ii + 1))
                        ratio2 = boundaries(ii) / rho(ii) * d(ii)
                        yc = yb(2 * ii + 1) + j * yb(n_eqns + 2 * ii + 1) &
                             - (yb(2 * ii - 1) + j * yb(n_eqns + 2 * ii - 1))
                        bcc = ratio2 * (yb(2 * ii) + j * yb(n_eqns + 2 * ii)) &
                            - sin(yc)
                        ! real part
                        bcb(2 * ii - 1) = yb(2 * ii) + ratio1 * yb(2 * ii + 2)
                        bcb(2 * ii) = bcc%re
                        ! imaginary part
                        bcb(n_layers + 2 * ii - 1) = yb(n_eqns + 2 * ii) &
                                + ratio1 * yb(n_eqns + 2 * ii + 2)
                        bcb(n_layers + 2 * ii) = bcc%im
                    end if
                ! Set the boundary conditions for even layers.
                else
                    ! The even layers only have left boundary conditions.
                    if (ii == n_layers) then
                        bca(ii) = ya(n_eqns)  ! real
                        bca(n_layers + ii) = ya(n_eqns + n_eqns)  ! imaginary
                    else
                        ratio1 = rho(ii) * d(ii) / (rho(ii + 1) * d(ii + 1))
                        ratio2 = boundaries(ii) / rho(ii) * d(ii)
                        yc = ya(2 * ii + 1) + j * ya(n_eqns + 2 * ii + 1) &
                             - (ya(2 * ii - 1) + j * ya(n_eqns + 2 * ii - 1))
                        bcc = ratio2 * (ya(2 * ii) + j * ya(n_eqns + 2 * ii)) &
                            + sin(yc)
                        ! real part
                        bca(2 * ii - 2) = ya(2 * ii) + ratio1 * ya(2 * ii + 2)
                        bca(2 * ii - 1) = bcc%re
                        ! imaginary part
                        bca(n_layers + 2 * ii - 2) = ya(n_eqns + 2 * ii) &
                                + ratio1 * ya(n_eqns + 2 * ii + 2)
                        bca(n_layers + 2 * ii - 1) = bcc%im
                    end if
                end if
            end do
        end subroutine bc
end subroutine solve_real
