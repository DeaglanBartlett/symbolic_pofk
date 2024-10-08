import torch
import math

from .linear import logF_fiducial as lcdm_logF_fiducial

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def growth_correction_R(theta_batch):
    """
    Correction to the growth factor for a batch of parameters.

    Args:
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :result (torch.Tensor): The correction to the growth factor for the given parameters
    """

    d = torch.tensor([0.8545, 0.394, 0.7294, 0.5347, 0.4662, 4.6669,
                      0.4136, 1.4769, 0.5959, 0.4553, 0.0799, 5.8311,
                      5.8014, 6.7085, 0.3445, 1.2498, 0.3756, 0.2136], device=device)

    # Unpack theta into individual parameters
    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)

    part1 = d[0]

    denominator_inner1 = a * \
        d[1] + d[2] + (Om * d[3] - a * d[4]) * \
        torch.log(-d[5] * w0 - d[6] * wa)
    part2 = -1 / denominator_inner1

    numerator_inner2 = Om * d[7] - a * d[8] + \
        torch.log(-d[9] * w0 - d[10] * wa)
    denominator_inner2 = -a * d[11] + d[12] + d[13] * \
        (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
    part3 = -numerator_inner2 / denominator_inner2

    result = 1 + (1 - a) * (part1 + part2 + part3)

    return result


def log10_S(k_batch, theta_batch):
    """
    Corrections to the present-day linear power spectrum for a batch of parameters.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :result (torch.Tensor): Corrections to the present-day linear power spectrum
    """

    e = torch.tensor([0.2841, 0.1679, 0.0534, 0.0024, 0.1183, 0.3971,
                      0.0985, 0.0009, 0.1258, 0.2476, 0.1841, 0.0316,
                      0.1385, 0.2825, 0.8098, 0.019, 0.1376, 0.3733], device=device)

    # Unpack theta into individual parameters
    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)

    part1 = -e[0] * h
    part2 = -e[1] * w0
    part3 = -e[2] * mnu / torch.sqrt(e[3] + k_batch**2)

    part4 = -(e[4] * h) / (e[5] * h + mnu)

    part5 = e[6] * mnu / (h * torch.sqrt(e[7] + (Om * e[8] + k_batch)**2))

    numerator_inner = (e[9] * Ob - e[10] * w0 - e[11] * wa +
                       (e[12] * w0 + e[13]) / (e[14] * wa + w0))
    denominator_inner = torch.sqrt(
        e[15] + (Om + e[16] * torch.log(-e[17] * w0))**2)

    part6 = numerator_inner / denominator_inner

    result = part1 + part2 + part3 + part4 + part5 + part6

    return result / 10


def get_approximate_D(k_batch, theta_batch):
    """
    Approximation to the growth factor using the results of
    Bond et al. 1980, Lahav et al. 1992, Carroll et al. 1992 
    and Eisenstein & Hu 1997 (D_cbnu).

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :D (torch.Tensor): Approximate linear growth factor at corresponding k values
    """

    # Unpack theta into individual parameters
    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)

    mnu = mnu + 1e-10

    z = 1 / a - 1
    theta2p7 = 2.7255 / 2.7
    zeq = 2.5e4 * Om * h ** 2 / theta2p7 ** 4

    Omega = Om * a ** (-3)
    OL = (1 - Om) * a ** (-3 * (1 + w0 + wa)) * torch.exp(-3 * wa * (1 - a))
    g = torch.sqrt(Omega + OL)
    Omega /= g ** 2
    OL /= g ** 2

    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2 /
        (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
    )

    Onu = mnu / 93.14 / h ** 2
    Oc = Om - Ob - Onu
    fc = Oc / Om
    fb = Ob / Om
    fnu = Onu / Om
    fcb = fc + fb

    pcb = 1/4 * (5 - torch.sqrt(1 + 24 * fcb))
    # Nnu = (3 if mnu != 0.0 else 0)
    Nnu = 3
    q = k_batch * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7/6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7/pcb) + (D1 / (1 + yfs)) **
             0.7) ** (pcb / 0.7) * D1 ** (1 - pcb)

    D = Dcbnu / (1 + zeq)

    return D


def get_eisensteinhu_nw(k_batch, theta_batch):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift zero for a batch of parameters.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :pk_batch (torch.Tensor): Approxmate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    # Unpack theta into individual parameters
    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)

    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7

    s = 44.5 * torch.log(9.83 / om0h2) / torch.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        torch.log(431.0 * om0h2) * ombom0 + 0.38 * \
        torch.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                      (1.0 + (0.43 * k_batch * h * s)**4))

    q = k_batch * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = torch.log(2.0 * math.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    pk_batch = (
        2 * torch.pi ** 2 / k_batch ** 3
        * (As * 1e-9) * (k_batch * h / kpivot) ** (ns - 1)
        * (2 * k_batch ** 2 * 2998**2 / 5 / Om) ** 2
        * tk_eh ** 2
    )

    return pk_batch


def logF_fiducial(k_batch, theta_batch):
    """
    Compute the emulated logarithm of the ratio between the true linear power spectrum 
    and the Eisenstein & Hu 1998 fit for LCDM modified from implementation in linear.py (Bartlett et al. 2023).

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :logF (torch.Tensor): The emulated logarithm of the ratio between the true linear power spectrum
    """

    logF = lcdm_logF_fiducial(k_batch, theta_batch[:, :5])

    return logF


def plin_plus_emulated(k, theta_batch):
    """
    Fiducial power spectrum given in Bartlett et al. 2023 for a batch of parameters.

    Args:
        :k (torch.Tensor): k values [h/Mpc] with shape (n_k)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9),
            the 9 parameters are :
                As: 10^9 times the amplitude of the primordial P(k)
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum
                mnu: Sum of neutrino masses [eV / c^2]
                w0: Time independent part of the dark energy EoS
                wa: Time dependent part of the dark energy EoS
                a: The scale factor to evaluate P(k) at

    Returns:
        :Pk (torch.Tensor): computed fiducial power spectrum for each k and theta in the batch
    """

    # add dim to k so that it can be broadcasted with theta_batch
    k_batch = k.unsqueeze(1).to(device)

    eh = get_eisensteinhu_nw(k_batch, theta_batch)
    D = get_approximate_D(k_batch, theta_batch)
    logF_value = logF_fiducial(k_batch, theta_batch)

    F_value = torch.exp(logF_value)
    R_value = growth_correction_R(theta_batch)
    log10_S_value = log10_S(k_batch, theta_batch)
    S_value = torch.pow(10, log10_S_value)

    Pk = eh * D ** 2 * F_value * R_value * S_value

    return Pk.T
