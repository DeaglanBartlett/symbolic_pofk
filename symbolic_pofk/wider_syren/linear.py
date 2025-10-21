import numpy as np


def symbolic_As(Omm, Omb, h, ns, sigma8):
    """
    Compute As from sigma8 and other cosmological parameters
    using a symbolic approximation

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    Returns:
        :As (float): The amplitude of the primordial power spectrum
    """
    
    b = [0.95534, 68.80781, 0.5159, 1.18861, 0.197, 0.53884, 0.01983, 0.76405, 0.29247, 10.88335, 
         0.73004, 1.20497, 0.75788, 2.14175, 3.03762, 4.71485, 5.46729, 0.9624]
    corr = (
        b[0]*ns*(b[1]*Omm)**(-b[2]*h)
        + (b[3]*np.sqrt(h) + b[4]*ns)*(-b[5]*ns + (Omb - b[6])/(b[7]*Omm - b[8]*Omb))
        + (b[9]*Omm)**(-b[10]*h)
        + (b[11]*Omm)**(b[12]*h)*(b[13]*Omb + (b[14]*h)**(-b[15]*Omm))/(b[16]*Omm + b[17]*Omb)
    )
    
    As = (sigma8 * np.exp(corr)) ** 2 * 1.e-9

    return As


def symbolic_sigma8(Omm, Omb, h, ns, As):
    """
    Compute sigma8 from As and other cosmological parameters
    using a symbolic approximation

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :As (float): The amplitude of the primordial power spectrum
    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """

    b = [0.95534, 68.80781, 0.5159, 1.18861, 0.197, 0.53884, 0.01983, 0.76405, 0.29247, 10.88335, 
         0.73004, 1.20497, 0.75788, 2.14175, 3.03762, 4.71485, 5.46729, 0.9624]
    corr = (
        b[0]*ns*(b[1]*Omm)**(-b[2]*h)
        + (b[3]*np.sqrt(h) + b[4]*ns)*(-b[5]*ns + (Omb - b[6])/(b[7]*Omm - b[8]*Omb))
        + (b[9]*Omm)**(-b[10]*h)
        + (b[11]*Omm)**(b[12]*h)*(b[13]*Omb + (b[14]*h)**(-b[15]*Omm))/(b[16]*Omm + b[17]*Omb)
    )

    sigma8 = np.sqrt(As * 1.e9) * np.exp(-corr)

    return sigma8


def symbolic_D(Omm, z):
    """
    Compute the linear growth factor for a LCDM universe at a given
    redshift using a symbolic approximations

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :z (float): Redshift to evaluate at

    Returns:
        :D (float): The linear growth factor at redshift z

    """

    a = 1/(1+z)
    x = (Omm - 1) / Omm * a ** 3

    b = [0.723, 2/3, 1.204]
    f = np.sqrt(b[2] + b[0]**b[1]) / np.sqrt((b[0] - x)**b[1] + b[2])

    D = a * f

    return D


def symbolic_f(Omm, z):
    """
    Compute the linear growth rate for a LCDM universe at a given
    redshift using a symbolic approximations

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :z (float): Redshift to evaluate at

    Returns:
        :f (float): The linear growth rate at redshift z

    """

    b = [0.8665, 0.6457, 0.9522]
    a = 1/(1+z)
    x = (Omm - 1) / Omm * a ** 3
    f = 1 + 3/2 * b[1] * x * (b[0] - x) ** (b[1] - 1) / \
        ((b[0] - x) ** b[1] + b[2])
    
    return f


def get_eisensteinhu_nw(Omm, Omb, h, ns, sigma8, z, k, TCMB=2.7255):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift z.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): Redshift to evaluate at
        :k (jnp.ndarray): k values to evaluate P(k) at [h / Mpc]
        :TCMB (float, default=2.7255): z=0 CMB Temperature [K]

    Returns:
        :pk (jnp.ndarray): Approximate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    # Cosmological parameters
    As = symbolic_As(Omm, Omb, h, ns, sigma8) * 1e9
    ombom0 = Omb / Omm
    om0h2 = Omm * h**2
    ombh2 = Omb * h**2
    theta2p7 = TCMB / 2.7

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        np.log(431.0 * om0h2) * ombom0 + 0.38 * \
        np.log(22.3 * om0h2) * ombom0**2
    Gamma = Omm * h * (alphaGamma + (1.0 - alphaGamma) /
                       (1.0 + (0.43 * k * h * s)**4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    #  Linear growth factor
    D = symbolic_D(Omm, z)

    pk = (
        2 * np.pi ** 2 / k ** 3
        * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
        * (2 * k ** 2 * 2998**2 / 5 / Omm) ** 2
        * tk_eh ** 2
        * D ** 2
    )

    return pk


def symbolic_pklin(Omm, Omb, h, ns, sigma8, z, k):
    """
    Compute the symbolic approximation to the linear P(k) at redshift z.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): Redshift to evaluate at
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]

    Returns:
        :pk (np.ndarray): Approximate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    c = [0.013924, 5.1771, 20.636, 0.49092, 3.4224, 0.35621, 0.016739, 0.18401, 0.58832, 5.108, 10.783,
         0.0043879, 0.11547, 1.3869, 2.9301, 32.014, 0.002192, 0.002926, 316.2, 1.2158, 0.095822,
         0.074921, 0.0067841, 0.0093912, 0.022678, 0.00093762, 0.64834, 0.006882, 0.6518, 0.11855,
         2.6863, 42.261, 121.6, 123.21, 14.51, 1378.1, 22.375, 1.0369, 75.439, 3.6282, 0.15279, 41.884,
         1052.3, 0.22687, 0.022003, 236.72, 0.16852, 0.55815, 0.040376, 0.4352, 0.011574, 0.043423, 0.001834,
         205.7, 0.26165, 0.0073544, 0.99135, 0.18444, 0.089257, 0.0074827, 7.7431e-05, 1.1788, 0.44205, 0.098723,
         0.01075, 0.007492, 0.82882, 1.1882, 383.17, 397.51, 4.4661, 0.81134, 0.0052959, 0.15939]

    # Compute raw SR fit
    F1 = (
        c[0]*(k + c[1]*(1-c[2]*k*Omm**(-c[3])) *
              (-Omm + (c[4] * Omb + c[5]*k)/np.sqrt(c[6] + (Omb + c[7]*k)**2)
               - c[8] * (c[9]*Omm)**(-c[10]*Omb)*k /
               np.sqrt((c[11]+h**2)*(c[12]*k**2 + Omm**2))
               + c[13]*(c[14]*Omm)**(-c[15]*Omb + c[16]*k)*k /
               np.sqrt(c[17] + (c[18]*Omm)**(c[19]*h)*(c[20]*Omm - k)**2)
               ) /
              (np.sqrt((c[21] + 1/(c[22] + (c[23] - c[24]*Omb - k)**2 /
                                    (c[25]+(k-c[26]*Omb)**2)))*(c[27] + (c[28]*Omb + c[29] - k)**2)))
              ))
    F2 = (1 / np.sqrt(1 +
                       ((c[30]*Omm)**(-c[31]*Omb)*(np.cos(c[32]*k) - np.cos(c[33]*k))/np.sqrt(1 + np.cos(c[34]*k)**2)
                        + c[35] * (c[36]*Omm)**(c[37]*h) * k**2 *
                        (c[38]*k)**(-c[39]*Omb) / np.sqrt(h**2 + c[40])
                        )**2))
    F3 = (
        (c[41]*(c[42]*Omm)**(c[43]*h)/np.sqrt(c[44] + h**2) -
         c[45]*((c[46]*Omm)**(c[47]*h*Omb - c[48]*h**2)))*k
        + c[49]/np.sqrt(c[50] + (c[51]*Omm - k)**2))
    F4 = (((c[52]*k*(c[53]*Omm)**(c[54]*h)/np.sqrt(c[55] + (c[56]*Omb - k)**2)/(Omb + c[57]*k))
           + (c[58]*k + c[59])/np.sqrt(c[60] + k**2*(c[61]*Omm)**(-c[62]*h)) - c[63] - c[64]/np.sqrt(k**2 + c[65]))
          / np.sqrt(1 + c[66]*(-Omm +
                                (c[67]*Omb + c[68]*Omm/Omb*(c[69]*Omm)**(-c[70]*h) + c[71]*k) /
                                np.sqrt(c[72] + (k - c[73])**2))**2))

    # Enforce low-k behaviour. F2->1 so we don't need a new variable for this
    F1_lowk = (c[0]*c[1]*(-Omm + c[4]*Omb/np.sqrt(c[6] + Omb**2)) /
               np.sqrt((c[21] + 1/(c[22] + (c[23]-c[24]*Omb)**2/(c[25]+(c[26]*Omb)**2)))*(c[27] + (c[28]*Omb + c[29])**2)))
    F3_lowk = c[49] / np.sqrt(c[50] + (c[51]*Omm)**2)
    F4_lowk = ((c[59]/np.sqrt(c[60]) - c[63] - c[64]/np.sqrt(c[65])) /
               np.sqrt(1 + c[66]*(-Omm + (c[67]*Omb + c[68]*Omm/Omb*(c[69]*Omm)**(-c[70]*h))/np.sqrt(c[72]+c[73]**2))**2))
    F5 = - F1_lowk * (np.cos(F3_lowk) + F4_lowk)

    # Combine fits
    logF = F1 * (F2 * np.cos(F3) + F4) + F5
    pk_nw = get_eisensteinhu_nw(Omm, Omb, h, ns, sigma8, z, k)
    pk = np.exp(logF) * pk_nw

    return pk
