
subroutine solve_diffusion_equation(energy, z, order, d, x_scale, n, m, theta)
    use bvp_m
    implicit none
    complex(kind=8), intent(in) :: energy
    integer, intent(in) :: n, m
    real(kind=8), intent(in) :: z(n), order(m, n), d(m), x_scale(m)
    complex(kind=8), intent(out) :: theta(m, n)


    theta(:, :) = 1D0
end