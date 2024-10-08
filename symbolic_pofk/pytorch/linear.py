import torch
import warnings
from .utils import simpson, hyp2f1

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def growth_factor(Om, acosmo):
    """
    Compute the growth factor D(a) normalised to a=1.

    Args:
        :Om (torch.Tensor): The z=0 total matter density parameter
        :acosmo (torch.Tensor): The scale factor.

    Returns:
        :Da (torch.Tensor): The normalised growth factor D(a).
    """
    a = 1.0 / 3.0
    b = 1.0
    c = 11.0 / 6.0

    # Compute z for D0
    z = (Om - 1.0) / Om
    D0 = hyp2f1(a, b, c, z)  # Hypergeometric function

    # Compute z for Da
    z = (Om - 1.0) / Om * acosmo**3
    Da = acosmo * hyp2f1(a, b, c, z)  # Hypergeometric function
    Da = Da / D0  # Normalize to D0

    return Da


def pk_EisensteinHu_zb(k, theta_batch, integral_norm=True):
    """
    Compute the Eisentein & Hu 1998 zero-baryon approximation to P(k) at z=0

    Args:
        :k (torch.Tensor): k values [h/Mpc] with shape (n_k)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 5 parameters are :
                sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
        :integral_norm (bool, default=True): Whether to compute the normalisation of the
            power spectrum using an integral over k

    Returns:
        :pk_eh (torch.Tensor): The Eisenstein & Hu 1998 zero-baryon P(k) [(Mpc/h)^3] for each
            k and theta in the batch
    """

    sigma8, Om, Ob, h, ns = theta_batch.unbind(dim=1)

    if integral_norm:
        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

        def get_pk(kk, Anorm):

            # Compute scale factor s, alphaGamma, and effective shape Gamma
            s = 44.5 * torch.log(9.83 / om0h2) / \
                torch.sqrt(1.0 + 10.0 * ombh2**0.75)
            alphaGamma = 1.0 - 0.328 * torch.log(431.0 * om0h2) * ombom0 + \
                0.38 * torch.log(22.3 * om0h2) * ombom0**2
            Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                              (1.0 + (0.43 * kk * h * s)**4))

            # Compute q, C0, L0, and tk_eh
            q = kk * theta2p7**2 / Gamma
            C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
            L0 = torch.log(
                2.0 * torch.exp(torch.tensor(1.0, device=device)) + 1.8 * q)
            tk_eh = L0 / (L0 + C0 * q**2)

            # Calculate Pk with unit amplitude
            return Anorm * tk_eh**2 * kk**ns

        # Define integration bounds and number of sub-intervals
        b0 = torch.log(torch.tensor(
            1e-7, dtype=torch.float32, device=device))  # ln(k_min)
        b1 = torch.log(torch.tensor(
            1e5, dtype=torch.float32, device=device))  # ln(k_max)
        # Number of sub-intervals (make sure it's even for Simpson's Rule)
        n = 1000

        # Find normalisation
        R = 8.0
        kk = torch.exp(torch.linspace(b0, b1, n))
        x = kk * R
        W = torch.zeros(x.shape)
        m = x < 1.e-3
        W[m] = 1.0
        W[~m] = 3.0 / x[~m]**3 * (torch.sin(x[~m]) - x[~m] * torch.cos(x[~m]))
        y = get_pk(kk, 1.0) * W**2 * kk**3
        sigma2 = simpson(y, x=torch.log(x))

        sigmaExact = torch.sqrt(sigma2 / (2.0 * torch.pi**2))
        Anorm = (sigma8 / sigmaExact)**2

        pk_eh = get_pk(k, Anorm)

    else:
        As = sigma8_to_As(theta_batch[:, :5])

        ombom0 = Ob / Om
        om0h2 = Om * h**2
        ombh2 = Ob * h**2
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

        # Compute scale factor s, alphaGamma, and effective shape Gamma
        s = 44.5 * torch.log(9.83 / om0h2) / \
            torch.sqrt(1.0 + 10.0 * ombh2**0.75)
        alphaGamma = 1.0 - 0.328 * \
            torch.log(431.0 * om0h2) * ombom0 + 0.38 * \
            torch.log(22.3 * om0h2) * ombom0**2
        Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                          (1.0 + (0.43 * k * h * s)**4))

        # Compute q, C0, L0, and tk_eh
        q = k * theta2p7**2 / Gamma
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        L0 = torch.log(
            2.0 * torch.exp(torch.tensor(1.0, device=device)) + 1.8 * q)
        tk_eh = L0 / (L0 + C0 * q**2)

        kpivot = 0.05

        pk_eh = (
            2 * torch.pi ** 2 / k ** 3
            * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
            * (2 * k ** 2 * 2998**2 / 5 / Om) ** 2
            * tk_eh ** 2
        )

        #  Get fitting formula without free-streaming
        a = 1.0
        z = 1 / a - 1
        theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin
        zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

        Omega = Om * a ** (-3)
        OL = (1 - Om)
        g = torch.sqrt(Omega + OL)
        Omega /= g ** 2
        OL /= g ** 2

        D1 = (
            (1 + zeq) / (1 + z) * 5 * Omega / 2 /
            (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
        )
        D1 /= (1 + zeq)
        pk_eh *= D1 ** 2

    return pk_eh


def logF_fiducial(k_batch, theta_batch, extrapolate=False, kmin=9.e-3, kmax=9):
    """
    Compute the emulated logarithm of the ratio between the true linear power spectrum 
    and the Eisenstein & Hu 1998 fit for LCDM as given in Bartlett et al. 2023.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 5 parameters are :
                sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for k < kmin and k > kmax
        :kmin (float, default=9.e-3): Minimum k value to use Bartlett et al. formula
            if extrapolate=False
        :kmax (float, default=9): Maximum k value to use Bartlett et al. formula
            if extrapolate=False

    Returns:
        :logF (torch.Tensor): The emulated logarithm of the ratio between the true linear power spectrum and the
            Eisenstein & Hu 1998 zero-baryon fit
    """

    b = torch.tensor([0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
                      4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
                      0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
                      0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
                      70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
                      4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
                      0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
                      7.385, 12.3960625361899, 0.0134114370723638], device=device)

    _, Om, Ob, h, _ = theta_batch.unbind(dim=1)

    line1 = b[0] * h - b[1]

    line2 = (
        ((Ob * b[2]) / torch.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k_batch - Ob) / torch.sqrt(b[6] + (Ob - b[7] * k_batch) ** 2)
            * b[8] * (b[9] * k_batch) ** (-b[10] * k_batch) * torch.cos(Om * b[11]
                                                                        - (b[12] * k_batch) / torch.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k_batch) /
                       torch.sqrt(1 + b[16] * k_batch ** 2) - Om)
            * torch.cos(b[17] * h / torch.sqrt(1 + b[18] * k_batch ** 2))
        )
    )

    line3 = (
        b[19] * (b[20] * Om + b[21] * h - torch.log(b[22] * k_batch)
                 + (b[23] * k_batch) ** (- b[24] * k_batch)) * torch.cos(b[25] / torch.sqrt(1 + b[26] * k_batch ** 2))
    )

    line4 = (
        (b[27] * k_batch) ** (-b[28] * k_batch) * (b[29] * k_batch - (b[30] * torch.log(b[31] * k_batch))
                                                   / torch.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * torch.cos(Om * b[34] - (b[35] * k_batch) / torch.sqrt(Ob ** 2 + b[36]))
    )

    logF = line1 + line2 + line3 + line4

    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k_batch >= kmin) & (k_batch <= kmax))
    if (not extrapolate) and m.sum() > 0:
        warnings.warn(
            "Not using Bartlett et al. formula outside tested regime")
        logF[m] = torch.log(
            pk_EisensteinHu_zb(k_batch[m], theta_batch, integral_norm=False) /
            pk_EisensteinHu_zb(k_batch[m], theta_batch, integral_norm=True)
        )

    return logF


def logF_max_precision(k, theta_batch, extrapolate=False, kmin=9.e-3, kmax=9):
    """
    Compute the emulated logarithm of the ratio between the true linear
    power spectrum and the Eisenstein & Hu 1998 fit. Here we use the mosy precide
    exprssion given in Bartlett et al. 2023.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 5 parameters are :
                sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for k < kmin and k > kmax
        :kmin (float, default=9.e-3): Minimum k value to use Bartlett et al. formula
            if extrapolate=False
        :kmax (float, default=9): Maximum k value to use Bartlett et al. formula
            if extrapolate=False

    Returns:
        :logF (torch.Tensor): The logarithm of the ratio between the linear P(k) and the
            Eisenstein & Hu 1998 zero-baryon fit
    """

    c = torch.tensor([5.143911, 0.867, 8.52, 0.292004226840437, 0.03101767374643,
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
                      1.28652398757532e-07, 2.04818021850729e-08], device=device)

    _, Om, Ob, h, _ = theta_batch.unbind(dim=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logF = (
            c[0]*k + c[1]*(c[2]*Ob - c[3]*k/torch.sqrt(k**2 + c[4]))*(-c[5]*(c[6]*Om
                                                                             - c[7]*k + (c[8]*Ob - c[9]*k)*torch.cos(c[10]*Om - c[11]*k))
                                                                      / ((c[12]*Om + c[13]*k)**(c[14]*k)*torch.sqrt((c[15]*Ob + k)**2 + c[16]))
                                                                      + c[17]*((c[18]*Ob + c[19]*Om - c[20]*h)*torch.cos(c[21]*Om - c[22]*k)
                                                                               + torch.cos(c[23]*k - c[24]))/((c[25]*k)**(c[26]*k)
                                                                                                              * torch.sqrt((-k + c[27]*(-c[28]*Om + c[29]*k)/torch.sqrt(k**2 + c[30]))**2
                                                                                                                           + c[31])))*(-torch.cos(c[32]*Om - c[33]*k)
                                                                                                                                       + c[34]/((c[35]*k)**(c[36]*k)*torch.sqrt((-Ob + c[37]*Om - c[38]*h)**2
                                                                                                                                                                                + c[39]))) - c[40]*(c[41]*Om - c[42]*h + c[43]*k + c[44]*k
                                                                                                                                                                                                    / (torch.sqrt(k**2 + c[45])*torch.sqrt((-Om - c[46]*h)**2 + c[47]))
                                                                                                                                                                                                    - c[48]*(c[49]*Om + c[50]*k)/torch.sqrt(k**2 + c[51]))
            * torch.cos(c[52]*k/(torch.sqrt(k**2 + c[53])*torch.sqrt((c[54]*Om - k)**2 + c[55])))
            - c[56] - c[57]*(c[58]*Om - c[59]*k + (-c[60]*Ob - c[61]*Om
                                                   + c[62]*h)*torch.cos(c[63]*Om - c[64]*k) + torch.cos(c[65]*k - c[66]))
            / ((c[67]*Om + c[68]*k)**(c[69]*k)*torch.sqrt(c[70]*(Ob + c[71]*h
                                                                 / torch.sqrt(k**2 + c[72]))**2/(k**2 + c[73]) + 1))
        )
    logF /= 100

    # Use Bartlett et al. 2023 P(k) only in tested regime
    m = ~((k >= kmin) & (k <= kmax))
    if (not extrapolate) and m.sum() > 0:
        warnings.warn(
            "Not using Bartlett et al. formula outside tested regime")
        logF[m] = torch.log(
            pk_EisensteinHu_zb(k[m], theta_batch, integral_norm=False) /
            pk_EisensteinHu_zb(k[m], theta_batch, integral_norm=True)
        )

    return logF


def plin_emulated(k, theta_batch, emulator='fiducial', extrapolate=False, kmin=9.e-3, kmax=9):
    """
    Compute the emulated linear matter power spectrum using the fits of
    Eisenstein & Hu 1998 and Bartlett et al. 2023.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 6),
            the 6 parameters are :
                sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                a: The scale factor to evaluate P(k) at
        :emulator (str, default='fiducial'): Which emulator to use from Bartlett et al.
            2023. 'fiducial' uses the fiducial one, and 'max_precision' uses the
            most precise one.
        :extrapolate (bool, default=False): If True, then extrapolates the Bartlett
            et al. 2023 fit outside range tested in paper. Otherwise, uses E&H with
            baryons for k < kmin and k > kmax
        :kmin (float, default=9.e-3): Minimum k value to use Bartlett et al. formula
            if extrapolate=False
        :kmax (float, default=9): Maximum k value to use Bartlett et al. formula
            if extrapolate=False

    Returns:
        :pk_lin (torch.Tensor): The emulated linear P(k) [(Mpc/h)^3]
    """

    p_eh = pk_EisensteinHu_zb(k, theta_batch[:, :5], integral_norm=True)
    if emulator == 'fiducial':
        logF = logF_fiducial(
            k, theta_batch[:, :5], extrapolate=extrapolate, kmin=kmin, kmax=kmax)
    elif emulator == 'max_precision':
        logF = logF_max_precision(
            k, theta_batch[:, :5], extrapolate=extrapolate, kmin=kmin, kmax=kmax)
    else:
        raise NotImplementedError
    p_lin = p_eh * torch.exp(logF)

    _, Om, _, _, _, a = theta_batch.unbind(dim=1)

    if torch.any(a != 1):
        D = growth_factor(Om, a)
        p_lin *= D ** 2

    return p_lin


def sigma8_to_As(theta_batch, old_equation=False):
    """
    Compute the emulated conversion sigma8 -> As as given in Bartlett et al. 2023

    Args:
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 5 parameters are :
                sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
        :old_equation (bool, default=False): Whether to use the version of the sigma8
            emulator which appeared in v1 of the paper on arXiv (True) or the final
            published version (and v2 on arXiv).

    Returns:
        :As (ftorch.Tensor): 10^9 times the amplitude of the primordial P(k)
    """

    sigma8, Om, Ob, h, ns = theta_batch.unbind(dim=1)

    if old_equation:
        a = torch.tensor([0.161320734729, 0.343134609906, -
                         7.859274, 18.200232, 3.666163, 0.003359], device=device)
        As = ((sigma8 - a[5]) / (a[2] * Ob + torch.log(a[3] * Om)) / torch.log(a[4] * h) -
              a[1] * ns) / a[0]
    else:
        a = torch.tensor([0.51172, 0.04593, 0.73983, 1.56738, 1.16846, 0.59348, 0.19994, 25.09218,
                          9.36909, 0.00011], device=device)
        f = (
            a[0] * Om + a[1] * h + a[2] * (
                (Om - a[3] * Ob)
                * (torch.log(a[4] * Om) - a[5] * ns)
                * (ns + a[6] * h * (a[7] * Ob - a[8] * ns + torch.log(a[9] * h)))
            )
        )
        As = (sigma8 / f) ** 2

    return As


def As_to_sigma8(theta_batch, old_equation=False):
    """
    Compute the emulated conversion As -> sigma8 as given in Bartlett et al. 2023

    Args:
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 5 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
        :old_equation (bool, default=False): Whether to use the version of the sigma8
            emulator which appeared in v1 of the paper on arXiv (True) or the final
            published version (and v2 on arXiv).

    Returns:
        :sigma8 (torch.Tensor): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """

    As, Om, Ob, h, ns = theta_batch.unbind(dim=1)

    if old_equation:
        a = torch.Tensor([0.161320734729, 0.343134609906, -
                         7.859274, 18.200232, 3.666163, 0.003359], device=device)
        sigma8 = (
            (a[0] * As + a[1] * ns) * (a[2] * Ob + torch.log(a[3] * Om))
            * torch.log(a[4] * h) + a[5]
        )
    else:
        a = torch.Tensor([0.51172, 0.04593, 0.73983, 1.56738, 1.16846, 0.59348, 0.19994, 25.09218,
                          9.36909, 0.00011], device=device)
        f = (
            a[0] * Om + a[1] * h + a[2] * (
                (Om - a[3] * Ob)
                * (torch.log(a[4] * Om) - a[5] * ns)
                * (ns + a[6] * h * (a[7] * Ob - a[8] * ns + torch.log(a[9] * h)))
            )
        )
        sigma8 = f * torch.sqrt(As)

    return sigma8
