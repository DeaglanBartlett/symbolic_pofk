import numpy as np
from .linear_new import plin_new_emulated


def pnl_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a):
    """
    Compute the non-linear power spectrum using a symbolic approximation
    to the linear power spectrum. 

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS
        :a (float): Scale factor to consider

    Returns:
        :pk_nl (np.ndarray): The emulated non-linear P(k) [(Mpc/h)^3]
    """

    g = np.array([0.2107, 0.0035, 0.0667, 0.0442, 1.2809, 0.2287, 0.1122, 4.3318, 1.1857, 3.3117, 14.2829,
                 0.9039, 0.0749, 0.0741, 0.1277, 27.6818, 24.8736, 0.6264, 0.3035, 0.6069, 0.7882, 0.4811,
                 1.4326, 1.8971, 0.0271, 0.9635, 0.0264, 22.9213, 71.1658, 0.0371, 0.0099, 210.3925, 0.2555])

    # calculate the linear power spectrum using the emulated model
    P_lin = np.log10(plin_new_emulated(k, As, Om, Ob, h, ns, mnu, w0, wa, a))

    term1 = P_lin

    numerator1 = g[0] * k * (g[1] * k)**(g[2] * Om - g[3] * As)
    denominator1_part1 = (g[4] * k**(-g[5]) - g[6] *
                          P_lin)**(g[7] * P_lin + g[8] * wa + g[9] * w0 - g[10])
    denominator1_part2 = (g[11] * k**g[12] + g[13] *
                          P_lin - g[14] * Om)**(g[15] * a - g[16] * ns)
    term2 = numerator1 / (denominator1_part1 + denominator1_part2)

    numerator2 = (g[17] * a - g[18] * P_lin + g[19]) * k
    denominator2 = (g[20] * Om + g[21] * k + g[22] * ns - g[23] +
                    (g[24] * P_lin + g[25] * k**g[26])**(g[27] * a - g[28] * ns))
    term3 = numerator2 / denominator2

    term4 = g[29] * k

    term5 = (g[30] * k)**((g[31] * k)**(-a * g[32]))

    # Combine all terms
    pk_nl = term1 + term2 + term3 - term4 - term5

    bias = pnl_bias(k)
    pk_nl = pk_nl - bias

    return np.power(10, pk_nl)


def pnl_bias(k):
    """
    the offset of the emulated non-linear power spectrum from euclidemulator2

    Parameters:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]

    Returns:
        :offset (np.ndarray): The offset of the emulated non-linear power spectrum
    """

    h = np.array([0.5787,   2.3485,  27.3829,  16.4236,  97.3766,  90.9764,
                  11.2046,  2447.2, 11376.93])

    term1 = ((h[1] * k) - np.cos(h[3]*np.cos(h[2] * k))) * np.cos(h[4] * k)
    denominator = -h[7]*np.log(h[6] * k) + (h[8] * k)

    offset = ((h[0] + term1 + np.cos(h[5] * k))) / denominator

    return offset
