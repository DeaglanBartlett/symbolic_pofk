import numpy as np
import symbolic_pofk.linear as linear


def ksigma_emulated(sigma8, Om, Ob, h, ns, a):
    """
    Emulate the non-linear scale of halofit using the approximation of
    Bartlett et al. 2024

    Args:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float): The scale factor

    Returns:
        :ksigma (float): The non-linear scale (h/Mpc)
    """

    c = [4.35761588, 0.83576576, 0.43023897, 20.107738, 0.259285, 0.573205,
         1.680897, 20.043272, 0.425699, 0.39078063]
    ksigma = (
        c[0] * (a * c[1] * (c[2] - sigma8)
                + (c[3] * a) ** (-c[4] * a - c[5] * ns)
                * (c[6] * Ob + (c[7] * Om) ** (-c[8] * h)))
        / (sigma8 * (a + c[9] * ns))
    )
    ksigma = np.exp(ksigma)

    return ksigma


def neff_emulated(sigma8, Om, Ob, h, ns, a):
    """
    Emulate the effective slope of the linear power spectrum at the non-linear scale
    using the approximation of Bartlett et al. 2024

    Args:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float): The scale factor

    Returns:
        :neff (float): The effective slope of the linear power spectrum at the
            non-linear scale
    """

    theta = [1.65139294e+00, 4.88150280e+00, 5.12499000e-01, 1.48848000e-01,
             1.56499400e+01, 2.39307000e-01, 1.34631000e-01]
    neff = (
        (theta[0] * ns - theta[1])*(theta[2] * Ob - theta[3] * h
                                    + (theta[4] * a) ** (-theta[5] * Om - theta[6] * sigma8))
    )

    return neff


def C_emulated(sigma8, Om, Ob, h, ns, a):
    """
    Emulate the curvature of the linear power spectrum at the non-linear scale
    using the approximation of Bartlett et al. 2024

    Args:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float): The scale factor

    Returns:
        :neff (float): The curvature of the linear power spectrum at the
            non-linear scale
    """

    b = [0.335853, 1.42946178682748, 0.115256188211481, 0.057211, 48.072159, 0.194058,
         1.176006, 1.015136, 0.235398, 0.359587, 2.389843, 0.356875, 0.443138]
    C = (
        b[0]*sigma8 - b[1]*np.sqrt(b[2]*ns + sigma8*(b[3]*h + (b[4]*Om)**(b[5]*a) -
                                                     b[6]))*(b[7]*Ob + b[8]*a + b[9]*sigma8 - (b[10]*h)**(b[11]*Om)) - b[12]
    )

    return C


def A_emulated(k, sigma8, Om, Ob, h, ns, a, ksigma=None, neff=None, C=None):
    """
    Compute the correction term to halofit as described in Bartlett et al. 2024

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float): The scale factor
        :ksigma (float, default=None): The non-linear scale (h/Mpc). If this is not
            provided, then we used the fitting formula of Bartlett et al. 2024
        :neff (float, default=None): The effective slope of the linear power spectrum
            at the non-linear scale. If this is not provided, then we used the fitting
            formula of Bartlett et al. 2024
        :C (float, default=None): The curvature of the linear power spectrum
            at the non-linear scale. If this is not provided, then we used the fitting
            formula of Bartlett et al. 2024

    Returns:
        :A (np.ndarray): The correction factor for halofit

    Returns:

    """

    if ksigma is None:
        ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, a)
    if neff is None:
        neff = neff_emulated(sigma8, Om, Ob, h, ns, a)
    if C is None:
        C = C_emulated(sigma8, Om, Ob, h, ns, a)
    y = k / ksigma

    d = [0.0, 0.2011, 1.2983, 16.8733, 3.6428, 1.0622, 0.1023, 2.2204,
         0.0105, 0.487, 0.6151, 0.3377, 3.315, 3.9819, 1.3572, 3.3259,
         0.3872, 4.1175, 2.6795, 5.3394, 0.0338]

    A = (d[0] - d[1] / np.sqrt(1 + (d[2] * y) ** (- d[3] * C))
         * (
         y - d[4] * (y - d[5] * ns) / np.sqrt((y - d[6] * np.log(d[7] * C)) ** 2
                                              + d[8]) + d[9] * neff / np.sqrt(d[10] + sigma8 ** 2) / np.sqrt((d[11] * y -
                                                                                                              np.cos(d[12] * neff)) ** 2 + 1)
         + (d[13] + d[14] * neff - d[15] * C - d[16] * y) * (d[17] * neff + d[18]
                                                             * y + np.cos(d[19] * neff)) / np.sqrt(y ** 2 + d[20])
         )
         )

    return A


def run_halofit(k, sigma8, Om, Ob, h, ns, a, emulator='fiducial', extrapolate=False, which_params='Bartlett', add_correction=True):
    """
    Compute the non-linear power spectrum using halofit and a symbolic approximation
    to the linear power spectrum. The model sr-halofit of Bartlett et al. 2024 is
    obtained with the default parameters.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float): The scale factor to evaluate P(k) at
        :emulator (str, default='fiducial'): Which linear P(k)emulator to use from
            Bartlett et al. 2023. 'fiducial' uses the fiducial one, and 'max_precision'
            uses the most precise one.
        :extrapolate (bool, default=False): If using the Bartlett et al. 2023 linear
            fit, then if true we extrapolate outside range tested in paper. Otherwise,
            we use the E&H with baryons fit for this regime. Due to problems with the Colossus
            Eisenstein & Hu fit on large scales (typically a 2 percent offset), it is
            strongly recommended to use extrapolate=False.
        :which_params (str, default='Bartlett'): Which halofit parameters to use.
            Currently available: 'Takahashi' uses those from Takahashi et al. 2012, or
            'Bartlett' uses those from Bartlett et al. 2024.
        :add_correction (bool, default=True): Whether to add the correction 1+A from
            Bartlett et al. 2024 or not.

    Returns:
        :pk_lin (np.ndarray): The emulated non-linear P(k) [(Mpc/h)^3]
    """

    if which_params == 'Bartlett':
        pars = [1.5358,  2.8533,  2.3692,  0.9916,  0.2244,  0.5862, -0.565,  0.5871,
                0.5757, -1.505,   0.3913,  2.0252,  0.7971,  0.5989,  0.2216, -0.001,
                1.1771,  5.2082, 3.7324, -0.0158, -0.0972,  0.155,   6.1043,  1.3408,
                -0.2138, -5.325,   1.9967, -0.7176,  0.3108,  1.2477,  0.4018, -0.3837,]
    elif which_params == 'Takahashi':
        pars = [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, 0.6083,
                -0.5642, 0.5864, 0.5716, -1.5474, 0.3698, 2.0404, 0.8161, 0.5869,
                0.1971, -0.0843, 0.8460, 5.2105, 3.6902, -0.0307, -0.0585,
                0.0743, 6.0835, 1.3373, -0.1959, -5.5274, 2.0379, -0.7354, 0.3157,
                1.2490, 0.3980, -0.1682]
    else:
        raise NotImplementedError

    ksigma = ksigma_emulated(sigma8, Om, Ob, h, ns, a)
    neff = neff_emulated(sigma8, Om, Ob, h, ns, a)
    C = C_emulated(sigma8, Om, Ob, h, ns, a)
    y = k / ksigma

    # Get linear P(k)
    plin = linear.plin_emulated(
        k, sigma8, Om, Ob, h, ns, a=a, emulator=emulator, extrapolate=extrapolate)

    # 1 halo term parameters
    an = (pars[0] + pars[1] * neff + pars[2] * neff ** 2 + pars[3] * neff ** 3
          + pars[4] * neff ** 4 - pars[5] * C)
    an = 10. ** an
    bn = pars[6] + pars[7] * neff + pars[8] * neff ** 2 + pars[9] * C
    bn = 10. ** bn
    cn = pars[10] + pars[11] * neff + pars[12] * neff ** 2 + pars[13] * C
    cn = 10. ** cn
    gamma = pars[14] + pars[15] * neff + pars[16] * C
    nu = 10. ** (pars[17] + pars[18] * neff)
    Omz = Om / a ** 3 / (Om / a ** 3 + 1. - Om)
    f1 = Omz ** pars[19]
    f2 = Omz ** pars[20]
    f3 = Omz ** pars[21]

    # 2 halo term parameters
    alpha = np.abs(pars[22] + pars[23] * neff +
                   pars[24] * neff ** 2 + pars[25] * C)
    beta = (pars[26] + pars[27] * neff + pars[28] * neff ** 2
            + pars[29] * neff ** 3 + pars[30] * neff ** 4 + pars[31] * C)

    # Predict 1 halo term
    deltaH2 = an * y ** (3 * f1) / (1 + bn * y ** f2 +
                                    (cn * f3 * y) ** (3 - gamma))
    deltaH2 /= 1 + nu / y ** 2
    ph = deltaH2 * (2 * np.pi ** 2) / k ** 3

    # Predict 2 halo term
    deltaL2 = k ** 3 * plin / (2 * np.pi ** 2)
    pq = plin * (1 + deltaL2) ** beta / \
        (1 + alpha * deltaL2) * np.exp(- y/4 - y**2/8)

    # Total prediction
    p_nl = ph + pq

    # Correction (1+A)
    if add_correction:
        p_nl *= 1 + A_emulated(k, sigma8, Om, Ob, h, ns, a,
                               ksigma=ksigma, neff=neff, C=C)

    return p_nl
