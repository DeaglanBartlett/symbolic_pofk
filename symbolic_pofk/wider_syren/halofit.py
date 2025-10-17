import numpy as np


def symbolic_ksigma(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the nonlinear scale used for halofit, ksigma.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :ksigma (float): The nonlinear scale for this cosmology
    """

    b = [0.3458, 0.01477, 0.0825, 4.642, 0.4738, 0.3847, 2.005, 0.02206, 0.2958, 0.4962, 0.03355, 0.6467, 1.139, 8.498,
         4.57, 0.6448, 0.1022, 0.3782, 1.239, 18.27, 0.1043, 0.3435, 0.2178, 0.1644, 0.2413, 0.03482, 0.5142, 0.4983,
         1.1, 0.8871, 0.7256, 0.1405, 0.614, 1.027, 3.036, 0.9132, 0.4545, 0.2444, 113.5, 97.35, 0.1534, 0.9639, 1.309, 1.616, 0.3708]

    F1 = -b[0]*sigma8 - (b[1]*h)**(b[2]*z*(b[3]*Omm) **
                                   (b[4]*sigma8 - b[5]*np.sqrt(z)))
    F2 = b[6]/h*(b[7]*z)**(b[8]*ns + b[9]*sigma8 - b[10]*z)
    F3 = (
        -b[11]*z*(-b[12]*Omb + (b[13]*Omm + np.log(b[14]*ns))
                  ** (-b[15]*z**(1-b[17]*h)*(b[16])**(-b[17]*h)))
        - (b[18]*sigma8*(b[19]*Omm)**(-b[20]*z) - (b[21]*ns)**((-b[22]*ns-b[23]*z)*(b[24]*h)**(b[25]*z))) *
        (-b[26]*h + b[27]*ns + (-b[28]*Omb + b[29]*Omm)
         ** (-b[30]*Omb - b[31]*sigma8))
    )
    F4 = (b[32]*Omm + b[33]*sigma8 - b[34] + (b[35]*ns)**(b[36]*ns + b[37]*z)
          + (-b[38]*Omb + b[39]*Omm)**(b[40]*sigma8) + (b[41]*ns)**(b[42]*sigma8*(b[43]*h)**(-b[44]*z)))

    ksigma = np.exp(F1 + F2 + F3/F4)

    return ksigma


def symbolic_neff(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the effective slope used for halofit, neff.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :neff (float): The effective slope for this cosmology
    """

    b = [3.3208, 6.3738, 0.2304, 0.1642, 0.064, 0.1461, 0.2171, 0.8835, 0.7457, 0.0537, 0.268,
         6.4778, 2.3502, 1.3872, 0.6122, 0.8784, 0.6466, 512.8273, 0.0894]

    neff = ((b[0]*np.sqrt(ns) - b[1])*(b[2]*ns + b[3]*sigma8 + b[4]*z - b[5])**(b[6]*sigma8)
            + (b[7]*np.sqrt(-Omb + b[8]*Omm) - b[9] /
               (h*(b[10]*np.sqrt(ns) + (b[11]*Omm)**(-b[12]*z - b[13]))**(b[14]*Omm)))
            * (b[15]*h + b[16]*sigma8 + (b[17]*z)**(-b[18]*z))
            )

    return neff


def symbolic_C(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the effective curvature used for halofit, C.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :C (float): The effective curvature for this cosmology
    """

    b = [4.917, 0.04262, 1461.0, 2181.0, 11.15, 0.4784, 0.09069, 0.0343, 0.04317, 0.0372, 0.09107, 0.151,
         0.04674, 0.04854, 0.05496, 0.03631, 0.1805, 0.1707, 0.2315, 0.4075, 0.593, 1.84, 1.028, 0.02645,
         0.06507, 0.06477, 0.192, 0.003867, 0.000556, 0.000851, 0.000177, 0.03328, 0.04181, 0.06002]

    C = (
        (b[0]*z)**(-b[1]*z) *
        (b[2]*Omm - b[3]*Omb + b[4]*z) **
        (b[5] + (b[6]*Omm)**(-b[7]*h+b[8]*ns-b[9]*sigma8) -
         (b[10]*z + b[11]*ns)**(-b[12]*Omm-b[13]*h-b[14]*sigma8-b[15]*z))
        * (-b[16]*h - b[17]*sigma8 + (b[18]*sigma8)**((b[19]*Omm-b[20]*Omb+b[21])**(-b[22]*h)*(b[23]*Omm + b[24]*h + b[25]*ns + b[26]*sigma8 - b[27]*z))
           - (-b[28]*Omm + b[29]*ns + b[30]*z)**(b[31]*h+b[32]*ns+b[33]*sigma8))
    )

    return C


def apply_halofit(k, plin, Omm, Omb, h, ns, sigma8, z, ksigma, neff, C):
    """
    Given a linear power spectrum, compute the halofit approximation to
    the nonlinear powerspectrum. The halofit variables ksigma, neff and C
    are computed using their symbolic approximations.

    Args:
        :k (jnp.ndarray): k values to evaluate P(k) at [h / Mpc]
        :plin (jnp.ndarray): The linear matter power spectrum [(Mpc/h)^3]
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): The redshift to evaluate P(k) at

    Returns:
        :p_nl (jnp.ndarray): The predicted non-linear P(k) [(Mpc/h)^3]
    """

    #  Takahasi et al. 2012 parameters
    pars = [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, 0.6083,
            -0.5642, 0.5864, 0.5716, -1.5474, 0.3698, 2.0404, 0.8161, 0.5869,
            0.1971, -0.0843, 0.8460, 5.2105, 3.6902, -0.0307, -0.0585,
            0.0743, 6.0835, 1.3373, -0.1959, -5.5274, 2.0379, -0.7354, 0.3157,
            1.2490, 0.3980, -0.1682]

    y = k / ksigma
    a = 1 / (1 + z)

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
    Omz = Omm / a ** 3 / (Omm / a ** 3 + 1. - Omm)
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

    return p_nl
