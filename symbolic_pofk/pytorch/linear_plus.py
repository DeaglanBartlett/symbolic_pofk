import torch
import math

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

    denominator_inner1 = a * d[1] + d[2] + (Om * d[3] - a * d[4]) * torch.log(-d[5] * w0 - d[6] * wa)
    part2 = -1 / denominator_inner1
    
    numerator_inner2 = Om * d[7] - a * d[8] + torch.log(-d[9] * w0 - d[10] * wa)
    denominator_inner2 = -a * d[11] + d[12] + d[13] * (Om * d[14] + a * d[15] - 1) * (d[16] * w0 + d[17] * wa + 1)
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
    denominator_inner = torch.sqrt(e[15] + (Om + e[16] * torch.log(-e[17] * w0))**2)

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
    Nnu =3
    q = k_batch * h * theta2p7 ** 2 / (Om * h ** 2)
    yfs = 17.2 * fnu * (1 + 0.488 / fnu ** (7/6)) * (Nnu * q / fnu) ** 2
    Dcbnu = (fcb ** (0.7/pcb) + (D1 / (1 + yfs)) ** 0.7) ** (pcb / 0.7) * D1 ** (1 - pcb)
    
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
    alphaGamma = 1.0 - 0.328 * torch.log(431.0 * om0h2) * ombom0 + 0.38 * torch.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * k_batch * h * s)**4))

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

def logF_fiducial(k_batch,theta_batch):
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
    
    b = torch.tensor([0.05448654, 0.00379, 0.0396711937097927, 0.127733431568858, 1.35,
        4.053543862744234, 0.0008084539054750851, 1.8852431049189666,
        0.11418372931475675, 3.798, 14.909, 5.56, 15.8274343004709, 0.0230755621512691,
        0.86531976, 0.8425442636372944, 4.553956000000005, 5.116999999999995,
        70.0234239999998, 0.01107, 5.35, 6.421, 134.309, 5.324, 21.532,
        4.741999999999985, 16.68722499999999, 3.078, 16.987, 0.05881491,
        0.0006864690561825617, 195.498, 0.0038454457516892, 0.276696018851544,
        7.385, 12.3960625361899, 0.0134114370723638], device=device)
    
    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)
        
    line1 = b[0] * h - b[1]
    
    line2 = (
        ((Ob * b[2]) / torch.sqrt(h ** 2 + b[3])) ** (b[4] * Om) *
        (
            (b[5] * k_batch - Ob) / torch.sqrt(b[6] + (Ob - b[7] * k_batch) ** 2)
            * b[8] * (b[9] * k_batch) ** (-b[10] * k_batch) * torch.cos(Om * b[11]
            - (b[12] * k_batch) / torch.sqrt(b[13] + Ob ** 2))
            - b[14] * ((b[15] * k_batch) / torch.sqrt(1 + b[16] * k_batch ** 2) - Om)
            * torch.cos(b[17] * h / torch.sqrt(1 + b[18] * k_batch ** 2))
        )
    )
    
    line3 = (
        b[19] *  (b[20] * Om + b[21] * h - torch.log(b[22] * k_batch)
        + (b[23] * k_batch) ** (- b[24] * k_batch)) * torch.cos(b[25] / torch.sqrt(1 + b[26] * k_batch ** 2))
    )
    
    line4 = (
        (b[27] * k_batch) ** (-b[28] * k_batch) * (b[29] * k_batch - (b[30] * torch.log(b[31] * k_batch))
        / torch.sqrt(b[32] + (Om - b[33] * h) ** 2))
        * torch.cos(Om * b[34] - (b[35] * k_batch) / torch.sqrt(Ob ** 2 + b[36]))
    )
    
    logF = line1 + line2 + line3 + line4

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

    F_value= torch.exp(logF_value)
    R_value = growth_correction_R(theta_batch)
    log10_S_value = log10_S(k_batch, theta_batch)
    S_value = torch.pow(10,log10_S_value)

    Pk = eh * D ** 2 * F_value * R_value * S_value

    return Pk.T
