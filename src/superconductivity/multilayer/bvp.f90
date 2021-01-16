subroutine solve_diffusion_equation(energy, z, order, d, rho, z_scale, m, n, theta)
    use bvp_m
    implicit none
    integer, parameter :: dp=kind(1d0)
    integer, intent(in) :: m, n
    real(kind=dp), intent(in) :: energy, z(n), order(m, n), d(m), rho(m), z_scale(m)
    complex(kind=dp), intent(out) :: theta(m, n)

    theta(:, :) = cmplx(m, n)

    contains
        function transform(z, i) result(zp)
            implicit none
            real(kind=dp), intent(in) :: z
            integer, intent(in) :: i
            real(kind=dp) :: zp

            if (mod(i, 2) == 1) then
                zp = (z - sum(d(1:i - 1))) / d(i)
            else
                zp = 1 - (z - sum(d(1:i - 1))) / d(i)
            end if
        end function transform


        function itransform(zp, i) result(z)
            implicit none
            real(kind=dp), intent(in) :: zp
            integer, intent(in) :: i
            real(kind=dp) :: z

            if (mod(i, 2) == 1) then
                z = d(i) * zp + sum(d(1:i - 1))
            else
                z = d(i) * (1 - zp) + sum(d(1:i - 1))
            end if
        end function itransform


        function delta(z) result(d)
            ! TODO: implement interpolation of order
        end function delta


        subroutine f(z, y, dydx)
            implicit none
            real(kind=dp), intent(in) :: z, y(n_eqns)
            real(kind=dp), intent(out) :: dydz(n_eqns)

            do i = 1, n_layers
                dydz(2 * i - 1) = y(2 * i)
                dydz(2 * i) = (z_scale)**2 * (energy * sin(y(2 * i - 1)) &
                                              + delta(z) * cos(y(2 * i - 1)))

            end do
        end subroutine f


        subroutine jac(z, y, dfdy)

        end subroutine jac


        subroutine bc(ya, yb, bca, bcb)

        end subroutine bc
end subroutine solve_diffusion_equation
