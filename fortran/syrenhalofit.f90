module constants
    implicit none
    real, parameter :: pi = 3.14159265358979323846
end module constants

module emulators
    implicit none
contains

    ! Function to compute E&H spectrum for a single k
    function scalar_eisenstein_hu_pk(k, Anorm, Om, Ob, h, ns) result(Pk)
        implicit none
        real, intent(in) :: k, Anorm, Om, Ob, h, ns
        real :: Pk, s, alphaGamma, Gamma, q, C0, L0, tk_eh, om0h2, ombh2, ombom0, theta2p7
        
        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7 ! Assuming Tcmb0 = 2.7255 Kelvin

        ! Compute scale factor s, alphaGamma, and effective shape Gamma
        s = 44.5 * log(9.83 / om0h2) / sqrt(1.0 + 10.0 * ombh2**0.75)
        alphaGamma = 1.0 - 0.328 * log(431.0 * om0h2) * ombom0 + 0.38 * log(22.3 * om0h2) * ombom0**2
        Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k * h * s)**4))

        ! Compute q, C0, L0, and tk_eh
        q = k * theta2p7**2 / Gamma
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        L0 = log(2.0 * exp(1.0) + 1.8 * q)
        tk_eh = L0 / (L0 + C0 * q**2)

        ! Calculate Pk
        Pk = Anorm * tk_eh**2 * k**ns
    end function scalar_eisenstein_hu_pk

    ! Function to compute sigma(R)
    function sigmaExact(R, Om, Ob, h, ns) result(sigma)
        use constants, only: pi
        implicit none
        real, intent(in) :: R, Om, Ob, h, ns
        real :: sigma, sigma2, a, b
        real :: x, hstep, sum_odd, sum_even, f_a, f_b
        integer :: n
        integer :: i

        ! Define integration bounds and number of sub-intervals
        a = log(1e-7) ! ln(k_min)
        b = log(1e5)  ! ln(k_max)
        n = 1000      ! Number of sub-intervals (make sure it's even for Simpson's Rule)

        if (mod(n, 2) /= 0) n = n + 1 ! Ensure n is even
        hstep = (b - a) / n
        sum_odd = 0.0
        sum_even = 0.0

        f_a = logIntegrand(a, R, Om, Ob, h, ns)
        f_b = logIntegrand(b, R, Om, Ob, h, ns)

        do i = 1, n - 1, 2
            x = a + i*hstep
            sum_odd = sum_odd + logIntegrand(x, R, Om, Ob, h, ns)
        end do

        do i = 2, n - 2, 2
            x = a + i*hstep
            sum_even = sum_even + logIntegrand(x, R, Om, Ob, h, ns)
        end do

        ! Use Simpson's Rule for integration
        sigma2 = hstep/3 * (f_a + 4*sum_odd + 2*sum_even + f_b)
        sigma = sqrt(sigma2 / (2.0 * pi**2))

    end function sigmaExact

    ! Function to compute integrand needed for sigmaExact
    function logIntegrand(lnk, R, Om, Ob, h, ns) result(ret)
        implicit none
        real, intent(in) :: lnk, R, Om, Ob, h, ns
        real :: ret, k, x, W, Pk

        k = exp(lnk)
        x = k * R
        if (x < 1e-3) then
            W = 1.0
        else
            W = 3.0 / x**3 * (sin(x) - x * cos(x))
        end if

        Pk = scalar_eisenstein_hu_pk(k, 1.0, Om, Ob, h, ns) ! Assumes Anorm=1.0 for this context
        ret = Pk * W**2 * k**3
    end function logIntegrand
    
    ! Function to correct the E&H ZB power spectrum
    function logF_fiducial(k, sigma8, Om, Ob, h, ns) result(logF)
        implicit none
	real, intent(in) :: k(:)
        real, intent(in) :: sigma8, Om, Ob, h, ns
        real, dimension(37) :: b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, &
            1.35, 4.053543862744234, 0.0008084539054750851, 1.8852431049189666, &
            0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691, &
            0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995, &
            70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532, &
            4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491, &
            0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544, &
            7.385, 12.3960625361899, 0.0134114370723638]
        real :: logF(size(k))
        
    logF = b(1) * h - b(2)
    
    logF = logF + (((Ob * b(3)) / sqrt(h ** 2 + b(4))) ** (b(5) * Om)  & 
        * ((b(6) * k - Ob) / sqrt(b(7) + (Ob - b(8) * k) ** 2) &
        * b(9) * (b(10) * k) ** (-b(11) * k) * cos(Om * b(12) &
        - (b(13) * k) / sqrt(b(14) + Ob ** 2)) &
        - b(15) * ((b(16) * k) / sqrt(1 + b(17) * k ** 2) - Om) &
        * cos(b(18) * h / sqrt(1 + b(19) * k ** 2))))
    
    logF = logF + (b(20) *  (b(21) * Om + b(22) * h - log(b(23) * k) &
        + (b(24) * k) ** (- b(25) * k)) * cos(b(26) / sqrt(1 + b(27) * k ** 2)))
    
    logF = logF + ((b(28) * k) ** (-b(29) * k) * (b(30) * k - (b(31) * log(b(32) * k)) &
        / sqrt(b(33) + (Om - b(34) * h) ** 2)) &
        * cos(Om * b(35) - (b(36) * k) / sqrt(Ob ** 2 + b(37))))
    
    end function logF_fiducial
    
    ! Series implementation for _2F_1
    function hypergeometric_series(a, b, c, z) result(F)
        implicit none
        real, intent(in) :: a, b, c, z
        real :: F, term, ap, bp, cp
        integer :: i, max_iter
        real, parameter :: tolerance = 1.0e-6
        
        if (abs(z) >= 1.0) then
            print *, "This implementation may not be accurate for |z| >= 1."
            F = 0.0
            return
        end if

        max_iter = 10000
        F = 1.0
        term = 1.0
        ap = a
        bp = b
        cp = c

        do i = 1, max_iter
            term = term * ap * bp / cp * z / real(i)
            F = F + term
            ap = ap + 1.0
            bp = bp + 1.0
            cp = cp + 1.0

            if (abs(term) < abs(F) * tolerance) exit
        end do
    
    end function hypergeometric_series
    
    ! Function to compute 2F1
    function hypergeometric_2F1(a, b, c, z) result(F)
        implicit none
        real, intent(in) :: a, b, c, z
        real :: F, new_z

        if (abs(z) < 1.0) then
            ! Use the direct series expansion for |z| < 1
            F = hypergeometric_series(a, b, c, z)
        else if (z > 1.0) then
            ! For real z > 1, indicate that this method is not applicable due to the branch cut
            print *, "The method is not applicable for real z > 1 due to the branch cut."
            F = 0.0
        else
            ! Apply Euler's transformation for |z| > 1
            new_z = z / (z - 1.0)
            F = (1.0 - z)**(-a) * hypergeometric_series(a, c-b, c, new_z)
        end if
    end function hypergeometric_2F1

    ! Function to compute D(a) normalised to a=1
    function growth_factor(Om, acosmo) result(Da)
        implicit none
        real, intent(in) :: Om, acosmo
        real :: D0, Da, a, b, c, z
        
        a = 1.0 / 3.0
        b = 1.0
        c = 11.0 / 6.0

	z = (Om - 1.0) / Om
        D0 = 1.0 * hypergeometric_2F1(a, b, c, z)

	z = (Om - 1.0) / Om * acosmo**3
        Da = acosmo * hypergeometric_2F1(a, b, c, z)
        Da = Da / D0
        
    end function growth_factor

    ! Function to calculate Eisenstein and Hu for vector k at z=0
    function eisenstein_hu_pk(k, sigma8, Om, Ob, h, ns) result(Pk)
        real, intent(in) :: sigma8, Om, Ob, h, ns
        real, intent(in) :: k(:)
        real :: Pk(size(k))
        integer :: i
        real :: Anorm

        ! Get normalisation to apply to E&H
        Anorm = (sigma8 / sigmaExact(8.0, Om, Ob, h, ns))**2
        
        ! Zero Baryon E&H
        do i = 1, size(k)
            Pk(i) = scalar_eisenstein_hu_pk(k(i), Anorm, Om, Ob, h, ns)
        end do

    end function eisenstein_hu_pk
    
    ! Function to calculate the Bartlett-correction linear power spectrum
    function linear_pk_emulated(k, sigma8, Om, Ob, h, ns, acosmo) result(Pk)
        real, intent(in) :: sigma8, Om, Ob, h, ns, acosmo
        real, intent(in) :: k(:)
        real :: Anorm, D
        real :: Pk(size(k))
        integer :: i
        
        ! Get normalisation to apply to E&H
        Anorm = (sigma8 / sigmaExact(8.0, Om, Ob, h, ns))**2
        
        ! Zero Baryon E&H
        do i = 1, size(k)
            Pk(i) = scalar_eisenstein_hu_pk(k(i), Anorm, Om, Ob, h, ns)
        end do
        
        ! Apply Bartlett+2023 correction
        Pk = Pk * exp(logF_fiducial(k, sigma8, Om, Ob, h, ns))
        
        ! Move to correct redshift
        D = growth_factor(Om, acosmo)
        Pk = Pk * D ** 2
    
    end function linear_pk_emulated

    ! Function to approximate As given sigma8 and other cosmological parameters
    ! This uses the conversion presented in v1 of Bartlett et al. which was
    ! changed in the final version. The final version function should be used
    ! as it is more accurate.
    function old_sigma8_to_as(sigma8, Om, Ob, h, ns) result(As)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns
        real, dimension(6) :: a = [0.161320734729, 0.343134609906, - 7.859274, &
                18.200232, 3.666163, 0.003359]
        real :: As

        As = ((sigma8 - a(6)) / (a(3) * Ob + log(a(4) * Om)) / log(a(5) * h) &
            - a(2) * ns) / a(1)
   
    end function old_sigma8_to_As

    ! Function to approximate sigma8 given As and other cosmological parameters
    ! This uses the conversion presented in v1 of Bartlett et al. which was
    ! changed in the final version. The final version function should be used
    ! as it is more accurate.
    function old_as_to_sigma8(As, Om, Ob, h, ns) result(sigma8)
        implicit none
        real, intent(in) :: As, Om, Ob, h, ns
        real, dimension(6) :: a = [0.161320734729, 0.343134609906, - 7.859274, &
            18.200232, 3.666163, 0.003359]
        real :: sigma8

        sigma8 = (a(1) * As + a(2) * ns) * (a(3) * Ob + log(a(4) * Om)) &
             * log(a(5) * h) + a(6)

    end function old_As_to_sigma8

    ! Function to approximate As given sigma8 and other cosmological parameters
    function sigma8_to_as(sigma8, Om, Ob, h, ns) result(As)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns
        real, dimension(10) :: a = [0.51172, 0.04593, 0.73983, 1.56738, 1.16846, &
            0.59348, 0.19994, 25.09218, 9.36909, 0.00011]
        real :: As

        As = (a(1) * Om + a(2) * h + a(3) * (Om - a(4) * Ob) &
            * (log(a(5) * Om) - a(6) * ns) &
            * (ns + a(7) * h * (a(8) * Ob - a(9) * ns + log(a(10) * h))))
        As = (sigma8 / As) ** 2

    end function sigma8_to_As

    ! Function to approximate sigma8 given As and other cosmological parameters
    function as_to_sigma8(As, Om, Ob, h, ns) result(sigma8)
        implicit none
        real, intent(in) :: As, Om, Ob, h, ns
        real, dimension(10) :: a = [0.51172, 0.04593, 0.73983, 1.56738, 1.16846, &
            0.59348, 0.19994, 25.09218, 9.36909, 0.00011]
        real :: sigma8

        sigma8 = (a(1) * Om + a(2) * h + a(3) * (Om - a(4) * Ob) &
            * (log(a(5) * Om) - a(6) * ns) &
            * (ns + a(7) * h * (a(8) * Ob - a(9) * ns + log(a(10) * h))))
        sigma8 = sigma8 * sqrt(As)

    end function As_to_sigma8

    ! Function to calculate ksigma
    function ksigma_emulated(sigma8, Om, Ob, h, ns, acosmo) result(ksigma)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns, acosmo
        real :: ksigma
        real, dimension(10) :: c = [4.35761588, 0.83576576, 0.43023897, &
            20.107738, 0.259285, 0.573205, 1.680897, 20.043272, 0.425699, 0.39078063]

        ksigma = (c(1) * (acosmo * c(2) * (c(3) - sigma8) + (c(4) * acosmo)**(-c(5) * acosmo - c(6) * ns) * &
            (c(7) * Ob + (c(8) * Om)**(-c(9) * h))) / (sigma8 * (acosmo + c(10) * ns)))
        ksigma = exp(ksigma)

    end function ksigma_emulated

    ! Function to calculate neff
    function neff_emulated(sigma8, Om, Ob, h, ns, acosmo) result(neff)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns, acosmo
        real :: neff
        real, dimension(7) :: theta = [1.65139294, 4.88150280, 0.512499, &
            0.148848, 15.64994, 0.239307, 0.134631]

        neff = (theta(1) * ns - theta(2)) * (theta(3) * Ob - theta(4) * h + &
            (theta(5) * acosmo)**(-theta(6) * Om - theta(7) * sigma8))

    end function neff_emulated

    ! Function to calculate C
    function C_emulated(sigma8, Om, Ob, h, ns, acosmo) result(C)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns, acosmo
        real :: C
        real, dimension(13) :: b = [0.335853, 1.42946178682748, 0.115256188211481, 0.057211, &
            48.072159, 0.194058, 1.176006, 1.015136, 0.235398, 0.359587, 2.389843, 0.356875, 0.443138]

        C = b(1)*sigma8 - b(2)*sqrt(b(3)*ns + sigma8*(b(4)*h + (b(5)*Om)**(b(6)*acosmo) - b(7)))* &
            (b(8)*Ob + b(9)*acosmo + b(10)*sigma8 - (b(11)*h)**(b(12)*Om)) - b(13)

    end function C_emulated
    
    function A_emulated(k, sigma8, Om, Ob, h, ns, acosmo, ksigma, neff, C) result(A)
        implicit none
        real, intent(in) :: sigma8, Om, Ob, h, ns, acosmo
        real, intent(in) :: k(:)
        real, optional, intent(in) :: ksigma, neff, C
        real :: A(size(k))
        real :: y, temp_ksigma, temp_neff, temp_C
        real, dimension(21) :: d = [0.0, 0.2011, 1.2983, 16.8733, 3.6428, &
            1.0622, 0.1023, 2.2204, 0.0105, 0.487, 0.6151, 0.3377, 3.315, &
            3.9819, 1.3572, 3.3259, 0.3872, 4.1175, 2.6795, 5.3394, 0.0338]
        integer :: i

        ! Determine if ksigma, neff, C are provided or need to be computed
        if (.not. present(ksigma)) then
            temp_ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, acosmo)
        else
            temp_ksigma = ksigma
        endif

        if (.not. present(neff)) then
            temp_neff = neff_emulated(sigma8, Om, Ob, h, ns, acosmo)
        else
            temp_neff = neff
        endif

        if (.not. present(C)) then
            temp_C = C_emulated(sigma8, Om, Ob, h, ns, acosmo)
        else
            temp_C = C
        endif

        ! Loop over k array to calculate A for each k
        do i = 1, size(k)
            y = k(i) / temp_ksigma
            A(i) = (d(1) - d(2) / sqrt(1.0 + (d(3) * y) ** (- d(4) * temp_C)) &
                * (y - d(5) * (y - d(6) * ns) / sqrt((y - d(7) * log(d(8) * temp_C)) ** 2 + d(9)) &
                + d(10) * temp_neff / sqrt(d(11) + sigma8 ** 2) / sqrt((d(12) * y - cos(d(13) * temp_neff)) ** 2 + 1.0) &
                + (d(14) + d(15) * temp_neff - d(16) * temp_C - d(17) * y) &
                * (d(18) * temp_neff + d(19) * y + cos(d(20) * temp_neff)) / sqrt(y ** 2 + d(21))))
        end do

    end function A_emulated
    
    function run_halofit(k, sigma8, Om, Ob, h, ns, acosmo, which_params, add_correction) result(p_pred)
        use constants, only: pi
        implicit none
        real, intent(in) :: k(:), sigma8, Om, Ob, h, ns, acosmo
        character(len=*), intent(in) :: which_params
        logical, intent(in) :: add_correction
        real :: p_pred(size(k))
        real :: ksigma, neff, C, y(size(k)), plin(size(k))
        real :: an, bn, cn, gamma, nu, Omz, f1, f2, f3
        real :: alpha, beta, deltaH2(size(k)), deltaL2(size(k)), ph(size(k)), pq(size(k))
        real, dimension(32) :: pars

        ! Select parameters based on 'which_params'
        if (which_params == 'Bartlett') then
            pars = [1.5358,  2.8533,  2.3692,  0.9916,  0.2244,  0.5862, -0.565,  0.5871,  0.5757, &
                 -1.505,   0.3913,  2.0252,  0.7971,  0.5989,  0.2216, -0.001,   1.1771,  5.2082, &
                  3.7324, -0.0158, -0.0972,  0.155,   6.1043,  1.3408, -0.2138, -5.325,   1.9967, &
                 -0.7176,  0.3108,  1.2477,  0.4018, -0.3837]
        else if (which_params == 'Takahashi') then
            pars = [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, 0.6083, &
              -0.5642, 0.5864, 0.5716, -1.5474, 0.3698, 2.0404, 0.8161, 0.5869, &
              0.1971, -0.0843, 0.8460, 5.2105, 3.6902, -0.0307, -0.0585, &
              0.0743, 6.0835, 1.3373, -0.1959, -5.5274, 2.0379, -0.7354, 0.3157, 1.2490, 0.3980, -0.1682]
        else
            ! Handle error or not implemented case
            stop 'NotImplementedError'
        end if

        ! Compute ksigma, neff, C, and y
        ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, acosmo)
        neff = neff_emulated(sigma8, Om, Ob, h, ns, acosmo)
        C = C_emulated(sigma8, Om, Ob, h, ns, acosmo)
        y = k / ksigma

        ! Get linear P(k)
        plin = linear_pk_emulated(k, sigma8, Om, Ob, h, ns, acosmo)

        ! Compute 1 halo term parameters...
        an = (pars(1) + pars(2) * neff + pars(3) * neff ** 2 + pars(4) * neff ** 3 &
            + pars(5) * neff ** 4 - pars(6) * C)
        an = 10. ** an
        bn = pars(7) + pars(8) * neff + pars(9) * neff ** 2 + pars(10) * C
        bn = 10. ** bn
        cn = pars(11) + pars(12) * neff + pars(13) * neff ** 2 + pars(14) * C
        cn = 10. ** cn
        gamma = pars(15) + pars(16) * neff + pars(17) * C
        nu = 10. ** (pars(18) + pars(19) * neff)
        Omz = Om / acosmo ** 3 / (Om / acosmo ** 3 + 1.- Om)
        f1 = Omz ** pars(20)
        f2 = Omz ** pars(21)
        f3 = Omz ** pars(22)
        
        ! 2 halo term parameters
        alpha = abs(pars(23) + pars(24) * neff + pars(25) * neff ** 2 + pars(26) * C)
        beta = (pars(27) + pars(28) * neff + pars(29) * neff ** 2 &
            + pars(30) * neff ** 3 + pars(31) * neff ** 4 + pars(32) * C)

        ! Predict 1 halo term
        deltaH2 = an * y ** (3 * f1) / (1 + bn * y ** f2 + (cn * f3 * y) ** (3 - gamma))
        deltaH2 = deltaH2 / (1 + nu / y ** 2)
        ph = deltaH2 * (2 * pi ** 2) / k ** 3 

        ! Predict 2 halo term
        deltaL2 = k ** 3 * plin/ (2 * pi ** 2)
        pq = plin * (1 + deltaL2) ** beta / (1 + alpha * deltaL2) * exp(- y / 4 - y ** 2 / 8)

        ! Total prediction
        p_pred = ph + pq

        ! Correction (1+A)
        if (add_correction) then
            p_pred = p_pred * (1.0 + A_emulated (k, sigma8, Om, Ob, h, ns, acosmo))
        end if

    end function run_halofit

end module emulators

program main
    use emulators
    implicit none
    integer, parameter :: N = 200
    real, dimension(N) :: k, Pk
    real :: k_min, k_max, log_k_min, log_k_max, delta
    integer :: i, niter, io_status
    integer, parameter :: unit_number = 10
    real :: sigma8, Om, Ob, h, ns, acosmo
    real :: start_time, end_time, mean_time_nocorr, mean_time_corr

    ! Initialize parameters
    sigma8 = 0.8
    Om = 0.3111
    Ob = 0.045
    h = 0.7
    ns = 0.96
    acosmo = 0.5
    k_min = 0.009
    k_max = 4.9
    
    ! Calculate log-spaced k values
    log_k_min = log10(k_min)
    log_k_max = log10(k_max)
    delta = (log_k_max - log_k_min) / (N - 1)
    do i = 1, N
        k(i) = 10.0**(log_k_min + (i - 1) * delta)
    end do

    ! Number of iterations
    niter = 1000

    call cpu_time(start_time)
    do i = 1, niter
        Pk = run_halofit(k, sigma8, Om, Ob, h, ns, acosmo, 'Bartlett', .true.)
    end do
    call cpu_time(end_time)
    mean_time_nocorr = (end_time - start_time) / real(niter)
    print *, "Mean CPU time (no correction): ", mean_time_nocorr, " seconds"
    
    call cpu_time(start_time)
    do i = 1, niter
        Pk = run_halofit(k, sigma8, Om, Ob, h, ns, acosmo, 'Bartlett', .true.)
    end do
    call cpu_time(end_time)
    mean_time_corr = (end_time - start_time) / real(niter)
    print *, "Mean CPU time (with correction): ", mean_time_corr, " seconds"
    
    ! Print to file
    open(unit=unit_number, file='output.txt', status='replace', action='write', iostat=io_status)
    write(unit_number, '("fortran_nocorr ", F25.17)') mean_time_nocorr
    write(unit_number, '("fortran_corr ", F25.17)') mean_time_corr
    close(unit=unit_number)
    
end program main
