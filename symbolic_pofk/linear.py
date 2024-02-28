import numpy as np
import warnings
import scipy.integrate
from colossus.cosmology import cosmology

def pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, use_colossus=False):
    """
    Compute the Eisentein & Hu 1998 zero-baryon approximation to P(k) at z=0
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :use_colossus (bool, default=True): Whether to use the external package colossus
            to compute this term
        
    Returns:
        :pk_eh (np.ndarray): The Eisenstein & Hu 1998 zero-baryon P(k) [(Mpc/h)^3]
    """

    if use_colossus:
        cosmo_params = {
            'flat':True,
            'sigma8':sigma8,
            'Om0':Om,
            'Ob0':Ob,
            'H0':h*100.,
            'ns':ns,
        }
        cosmo = cosmology.setCosmology('myCosmo', **cosmo_params)
        pk_eh = cosmo.matterPowerSpectrum(k, z = 0.0, model='eisenstein98_zb')
    else:
        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7 # Assuming Tcmb0 = 2.7255 Kelvin

        def get_pk(kk, Anorm):
        
            # Compute scale factor s, alphaGamma, and effective shape Gamma
            s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
            alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + \
            0.38 * np.log(22.3 * om0h2) * ombom0**2
            Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / \
                (1.0 + (0.43 * kk * h * s)**4))
            
            # Compute q, C0, L0, and tk_eh
            q = kk * theta2p7**2 / Gamma
            C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
            L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
            tk_eh = L0 / (L0 + C0 * q**2)

            # Calculate Pk with unit amplitude
            return Anorm * tk_eh**2 * kk**ns
        
        # Define integration bounds and number of sub-intervals
        b0 = np.log(1e-7) # ln(k_min)
        b1 = np.log(1e5)  # ln(k_max)
        n = 1000      # Number of sub-intervals (make sure it's even for Simpson's Rule)

        # Find normalisation
        R = 8.0
        kk = np.exp(np.linspace(b0, b1, n))
        x = kk * R
        W = np.zeros(x.shape)
        m = x < 1.e-3
        W[m] = 1.0
        W[~m] =3.0 / x[~m]**3 * (np.sin(x[~m]) - x[~m] * np.cos(x[~m]))
        y = get_pk(kk, 1.0) * W**2 * kk**3
        sigma2 = scipy.integrate.simpson(y, x=np.log(x))
        
        sigmaExact = np.sqrt(sigma2 / (2.0 * np.pi**2))
        Anorm = (sigma8 / sigmaExact)**2
        
        pk_eh = get_pk(k, Anorm)
        
    return pk_eh
    
    
def pk_EisensteinHu_b(k, sigma8, Om, Ob, h, ns):
    """
    Compute the Eisentein & Hu 1998 baryon approximation to P(k) at z=0
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        
    Returns:
        :pk_eh (np.ndarray): The Eisenstein & Hu 1998 baryon P(k) [(Mpc/h)^3]
    """

    cosmo_params = {
        'flat':True,
        'sigma8':sigma8,
        'Om0':Om,
        'Ob0':Ob,
        'H0':h*100.,
        'ns':ns,
    }
    cosmo = cosmology.setCosmology('myCosmo', **cosmo_params)
    pk_eh = cosmo.matterPowerSpectrum(k, z = 0.0, model='eisenstein98')
        
    return pk_eh
    

def logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=False):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the fiducial exprssion
    given in Bartlett et al. 2023.
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for this regime
        
    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """
    
    b = [0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
        4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
        0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
        0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
        70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
        4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
        0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
        7.385, 12.3960625361899, 0.0134114370723638]
        
    line1 = b[0] * h - b[1]
    
    line2 = (
        ((Ob * b[2]) / np.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k - Ob) / np.sqrt(b[6] + (Ob - b[7] * k) ** 2)
            * b[8] * (b[9] * k) ** (-b[10] * k) * np.cos(Om * b[11]
            - (b[12] * k) / np.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k) / np.sqrt(1 + b[16] * k ** 2) - Om)
            * np.cos(b[17] * h / np.sqrt(1 + b[18] * k ** 2))
        )
    )
    
    line3 = (
        b[19] *  (b[20] * Om + b[21] * h - np.log(b[22] * k)
        + (b[23] * k) ** (- b[24] * k)) * np.cos(b[25] / np.sqrt(1 + b[26] * k ** 2))
    )
    
    line4 = (
        (b[27] * k) ** (-b[28] * k) * (b[29] * k - (b[30] * np.log(b[31] * k))
        / np.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * np.cos(Om * b[34] - (b[35] * k) / np.sqrt(Ob ** 2 + b[36]))
    )
    
    logF = line1 + line2 + line3 + line4
    
    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k >= 9.e-3) & (k <= 9))
    if (not extrapolate) and m.sum() > 0:
        warnings.warn("Not using Bartlett et al. formula outside tested regime")
        logF[m] = np.log(
            pk_EisensteinHu_b(k[m], sigma8, Om, Ob, h, ns) /
            pk_EisensteinHu_zb(k[m], sigma8, Om, Ob, h, ns)
        )

    return logF
    
    
def logF_max_precision(k, sigma8, Om, Ob, h, ns, extrapolate=False):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the mosy precide
    exprssion given in Bartlett et al. 2023.
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for this regime
        
    Returns:
        :logF (np.ndarray): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """
    
    c = [5.143911, 0.867, 8.52, 0.292004226840437, 0.03101767374643,
        0.00329011222572802, 240.234, 682.449, 2061.023, 6769.493, 7.125, 108.136,
        6.2, 2.882, 59.585, 0.138395280663023, 1.0824838457885e-05,
        0.00328579877768286, 85.791, 19.855, 15.939, 9.547, 97.34, 94.83, 1.881, 3.945,
        11.151, 0.000423716845242328, 26.822, 230.12, 0.000854664012107594,
        1.07964736074221e-05, 3.162, 99.918, 0.12102142079148, 0.495, 12.091,
        0.607043446690064, 0.0176691274355561, 0.0146461842903885, 0.867, 32.371,
        7.058, 6.075, 16.3109603575571, 0.0024507401235173, 0.163242642976124,
        0.0770748313371466, 0.0522056904202558, 22.722, 774.688, 0.00272543411225559,
         1.03366440501996, 0.00582005561365782, 0.247172859450727, 0.289985439831066,
         0.241, 0.867, 2.618, 2.1, 114.391, 13.968, 11.133, 4.205, 100.376, 106.993,
         3.359, 1.539, 1.773, 18.983, 0.383761383928493, 0.00238796278978367,
         1.28652398757532e-07, 2.04818021850729e-08]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logF =(
            c[0]*k + c[1]*(c[2]*Ob - c[3]*k/np.sqrt(k**2 + c[4]))*(-c[5]*(c[6]*Om
            - c[7]*k + (c[8]*Ob - c[9]*k)*np.cos(c[10]*Om - c[11]*k))
            /((c[12]*Om + c[13]*k)**(c[14]*k)*np.sqrt((c[15]*Ob + k)**2 + c[16]))
            + c[17]*((c[18]*Ob + c[19]*Om - c[20]*h)*np.cos(c[21]*Om - c[22]*k)
            + np.cos(c[23]*k - c[24]))/((c[25]*k)**(c[26]*k)
            *np.sqrt((-k + c[27]*(-c[28]*Om + c[29]*k)/np.sqrt(k**2 + c[30]))**2
            + c[31])))*(-np.cos(c[32]*Om - c[33]*k)
            + c[34]/((c[35]*k)**(c[36]*k)*np.sqrt((-Ob + c[37]*Om - c[38]*h)**2
            + c[39]))) - c[40]*(c[41]*Om - c[42]*h + c[43]*k + c[44]*k
            /(np.sqrt(k**2 + c[45])*np.sqrt((-Om - c[46]*h)**2 + c[47]))
            - c[48]*(c[49]*Om + c[50]*k)/np.sqrt(k**2 + c[51]))
            *np.cos(c[52]*k/(np.sqrt(k**2 + c[53])*np.sqrt((c[54]*Om - k)**2 + c[55])))
            - c[56] - c[57]*(c[58]*Om - c[59]*k + (-c[60]*Ob - c[61]*Om
            + c[62]*h)*np.cos(c[63]*Om - c[64]*k) + np.cos(c[65]*k - c[66]))
            /((c[67]*Om + c[68]*k)**(c[69]*k)*np.sqrt(c[70]*(Ob + c[71]*h
            /np.sqrt(k**2 + c[72]))**2/(k**2 + c[73]) + 1))
        )
    logF /= 100
    
    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k >= 9.e-3) & (k <= 9))
    if (not extrapolate) and m.sum() > 0:
        warnings.warn("Not using Bartlett et al. formula outside tested regime")
        logF[m] = np.log(
            pk_EisensteinHu_b(k[m], sigma8, Om, Ob, h, ns) /
            pk_EisensteinHu_zb(k[m], sigma8, Om, Ob, h, ns)
        )

    return logF
    
    
def plin_emulated(k, sigma8, Om, Ob, h, ns, a=1, emulator='fiducial',
    extrapolate=False):
    """
    Compute the emulated linear matter power spectrum using the fits of
    Eisenstein & Hu 1998 and Bartlett et al. 2023.
    
    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :a (float, default=1): The scale factor to evaluate P(k) at
        :emulator (str, default='fiducial'): Which emulator to use from Bartlett et al.
            2023. 'fiducial' uses the fiducial one, and 'max_precision' uses the
            most precise one.
        :extrapolate (bool, default=False): If using the Bartlett et al. 2023 fit, then
            if true we extrapolate outside range tested in paper. Otherwise, we use the
            E&H with baryons fit for this regime
        
    Returns:
        :pk_lin (np.ndarray): The emulated linear P(k) [(Mpc/h)^3]
    """
    
    p_eh = pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)
    if emulator == 'fiducial':
        logF = logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=extrapolate)
    elif emulator == 'max_precision':
        logF = logF_max_precision(k, sigma8, Om, Ob, h, ns, extrapolate=extrapolate)
    else:
        raise NotImplementedError
    p_lin = p_eh * np.exp(logF)
    
    if a != 1:
        # Get growth factor
        cosmo_params = {
            'flat':True,
            'sigma8':sigma8,
            'Om0':Om,
            'Ob0':Ob,
            'H0':h*100.,
            'ns':ns,
        }
        cosmo = cosmology.setCosmology('myCosmo', **cosmo_params)
        D = cosmo.growthFactor(1/a - 1)
        
        # Linear P(k) at z
        p_lin *= D ** 2
        
    return p_lin


def sigma8_to_As(sigma8, Om, Ob, h, ns):
    """
    Compute the emulated conversion sigma8 -> As as given in Bartlett et al. 2023
    
    Args:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        
    Returns:
        :As (float): 10^9 times the amplitude of the primordial P(k)
    """
    
    a = [0.161320734729, 0.343134609906, - 7.859274, 18.200232, 3.666163, 0.003359]
    As = ((sigma8 - a[5]) / (a[2] * Ob + np.log(a[3] * Om)) / np.log(a[4] * h) -
        a[1] * ns) / a[0]
    
    return As
    
def As_to_sigma8(As, Om, Ob, h, ns):
    """
    Compute the emulated conversion As -> sigma8 as given in Bartlett et al. 2023
    
    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        
    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """
    
    a = [0.161320734729, 0.343134609906, - 7.859274, 18.200232, 3.666163, 0.003359]
    sigma8 = (
        (a[0] * As + a[1] * ns) * (a[2] * Ob + np.log(a[3] * Om))
        * np.log(a[4] * h) + a[5]
    )

    return sigma8
