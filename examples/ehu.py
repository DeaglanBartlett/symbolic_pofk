import numpy as np
import camb
import matplotlib.pyplot as plt


def get_eisensteinhu(k, As, Om, Ob, h, ns):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift zero.

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum

    Returns:
        :pk (np.ndarray): Approxmate linear power spectrum at corresponding k values [(Mpc/h)^3]

    """

    ombom0 = Ob / Om
    om0h2 = Om * h**2
    ombh2 = Ob * h**2
    theta2p7 = 2.7255 / 2.7  # Assuming Tcmb0 = 2.7255 Kelvin

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        np.log(431.0 * om0h2) * ombom0 + 0.38 * \
        np.log(22.3 * om0h2) * ombom0**2
    Gamma = Om * h * (alphaGamma + (1.0 - alphaGamma) /
                      (1.0 + (0.43 * k * h * s)**4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    pk = (
        2 * np.pi ** 2 / k ** 3
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
    g = np.sqrt(Omega + OL)
    Omega /= g ** 2
    OL /= g ** 2

    D1 = (
        (1 + zeq) / (1 + z) * 5 * Omega / 2 /
        (Omega ** (4/7) - OL + (1 + Omega/2) * (1 + OL/70))
    )
    D1 /= (1 + zeq)
    pk *= D1 ** 2

    return pk


def get_camb_linear(k, As, Om, Ob, h, ns):
    """
    Compute linear P(k) using camb for the cosmology of interest

    Args:
        :k (np.ndarray): k values to evaluate P(k) at [h / Mpc]
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Omega_m
        :Ob (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum

    Returns:
        :plin_camb (np.ndarray): Linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    tau = 0.0561

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=0.0,
                       omk=0,
                       tau=tau,)
    pars.set_matter_power(redshifts=[0.], kmax=k[-1])
    pars.NonLinear = camb.model.NonLinear_none
    pars.InitPower.set_params(As=As*1e-9, ns=ns, r=0)
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
        minkh=k.min(), maxkh=k.max(), npoints=len(k))
    pk_camb = pk_camb[0, :]

    return pk_camb


def main():

    kmin = 1e-4
    kmax = 1e0
    nk = 200

    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    As = 2.105  # 10^9 A_s
    h = 0.6766
    Om = 0.3111
    Ob = 0.02242 / h ** 2
    ns = 0.9665

    pk_eh = get_eisensteinhu(k, As, Om, Ob, h, ns)
    pk_camb = get_camb_linear(k, As, Om, Ob, h, ns)

    plt.semilogx(k, pk_eh/pk_camb, label='Eisenstein & Hu')
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    main()
    quit()
