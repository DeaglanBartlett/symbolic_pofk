import torch


def syren_baryon_Astrid(k, z, Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2):
    """"
    Compute the impact of baryonic physics on the matter power spectrum.

    The explanation of astrophysical parameters was taken from https://camels.readthedocs.io/en/latest/parameters.html

    Args:
        :k (Union[float, np.ndarray]): Wavenumber in unit k / h Mpc-1
        :z (Union[float, np.ndarray]): Redshift
        :Omega_m (Union[float, np.ndarray]): Density of matter.
        :sigma_8 (Union[float, np.ndarray]): Amplitude of matter fluctuations
        :A_SN1 (Union[float, np.ndarray]): Energy per SFR of the galactic winds
        :A_AGN1 (Union[float, np.ndarray]): Energy per black-hole accretion rate of the kinetic black-hole feedback
        :A_SN2 (Union[float, np.ndarray]): Wind speed of the galactic winds
        :A_AGN2 (Union[float, np.ndarray]): Energy per unit black-hole accretion rate of the thermal model of the black-hole feedback

    Returns:
        :S (Union[float, np.ndarray)): Baryonic suppression of the matter power spectrum
    """

    alpha_1 = 7.9*A_SN2/Omega_m
    alpha_2 = 0.0014
    alpha_3 = 0.937*A_SN2
    alpha_4 = 1.622*A_AGN2 + 0.849*A_SN1 + 5.092*A_SN2**2 * \
        (0.23*A_AGN1 - A_AGN2 - 0.71*A_SN1 + 3.107*sigma_8)
    alpha_5 = 0.78
    alpha_6 = 0.677
    alpha_7 = A_AGN2*(19.756*Omega_m + 2.8478)
    alpha_8 = 1.7347*A_AGN2 + 11.389*Omega_m + 1.642
    alpha_9 = 0.0224
    alpha_10 = 0.029
    alpha_11 = 0.063
    log_s = alpha_1*k*(alpha_2*k)**alpha_3*(alpha_4 - z)/((alpha_5*z - 1/alpha_6**(2*z))*(
        alpha_7 + alpha_8*k + k**2)) + (alpha_9*k - 1 + torch.exp(-alpha_10*k))*torch.exp(-alpha_11*z)
    s = torch.exp(log_s)

    return s


def syren_baryon_IllustrisTNG(k, z, Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2):
    """"
    Compute the impact of baryonic physics on the matter power spectrum.

    The explanation of astrophysical parameters was taken from https://camels.readthedocs.io/en/latest/parameters.html

    Args:
        :k (Union[float, np.ndarray]): Wavenumber in unit h Mpc-1
        :z (Union[float, np.ndarray]): Redshift
        :Omega_m (Union[float, np.ndarray]): Density of matter.
        :sigma_8 (Union[float, np.ndarray]): Amplitude of matter fluctuations
        :A_SN1 (Union[float, np.ndarray]): Energy per unit SFR of the galactic winds
        :A_AGN1 (Union[float, np.ndarray]): Energy per unit blach-hole accretion rate, not used in the model, only here for consistency with other simulators.
        :A_SN2 (Union[float, np.ndarray]): Wind speed of the galactic winds
        :A_AGN2 (Union[float, np.ndarray]): Ejection speed/burstiness of the kinetic mode of the black-hole feedback

    Returns:
        :S (Union[float, np.ndarray)): Baryonic suppression of the matter power spectrum
    """

    alpha_1 = 0.0109*A_SN2/Omega_m
    alpha_2 = 3592.322*Omega_m
    alpha_3 = 0.0087
    alpha_4 = 0.059
    alpha_5 = (0.9007*A_AGN2 - 0.5901*A_SN1 -
               1.5576*A_SN2 + 2.6846*sigma_8)/A_SN2
    alpha_6 = 0.048
    alpha_7 = 0.79265536265842*Omega_m**0.193
    alpha_8 = 0.022
    alpha_9 = 0.021
    alpha_10 = 0.797
    log_s = -alpha_1*alpha_2**(-alpha_3*k - alpha_4*z)*k*(alpha_5*(-alpha_6 *
                                                                   k + alpha_7**z) + 1) + alpha_8*torch.exp(-alpha_9*torch.exp(alpha_10*z)/k)
    s = torch.exp(log_s)

    return s


def syren_baryon_SIMBA(k, z, Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2):
    """"
    Compute the impact of baryonic physics on the matter power spectrum.

    The explanation of astrophysical parameters was taken from https://camels.readthedocs.io/en/latest/parameters.html

    Args:
        :k (Union[float, np.ndarray]): Wavenumber in unit h Mpc-1
        :z (Union[float, np.ndarray]): Redshift
        :Omega_m (Union[float, np.ndarray]): Density of matter.
        :sigma_8 (Union[float, np.ndarray]): Amplitude of matter fluctuations
        :A_SN1 (Union[float, np.ndarray]): Mass loading of the galactic winds
        :A_AGN1 (Union[float, np.ndarray]): Momentum flux of the QSO & jet-mode black-hole feedback
        :A_SN2 (Union[float, np.ndarray]): Wind speed of the galactic winds
        :A_AGN2 (Union[float, np.ndarray]): Jet speed of the jet-mode black-hole feedback

    Returns:
        :S (Union[float, np.ndarray)): Baryonic suppression of the matter power spectrum
    """

    alpha_1 = 0.00133/Omega_m**2
    alpha_2 = Omega_m*(25.727*A_AGN1 + 153.382*A_AGN2 -
                       70.6364*A_SN1 + 260.812*sigma_8)
    alpha_3 = 0.0553
    alpha_4 = 0.79055
    alpha_5 = 0.6024
    alpha_6 = 1.594
    alpha_7 = 16.01
    alpha_8 = 12.0685*(1.0047*A_SN2)**(0.6788*A_SN1)
    alpha_9 = 4.4661
    alpha_10 = 114.529
    alpha_11 = 1.21
    log_s = -alpha_1*k*(alpha_2 + k)/((alpha_5*z + (alpha_3*k)**(alpha_4*z))*(alpha_7*z +
                                                                              alpha_8 + k**alpha_6)) + alpha_9*torch.exp(-alpha_10*torch.exp(-alpha_11*z)/k - 2*z)
    s = torch.exp(log_s)

    return s


def syren_baryon_Swift_EAGLE(k, z, Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2):
    """"
    Compute the impact of baryonic physics on the matter power spectrum.

    The explanation of astrophysical parameters was taken from https://camels.readthedocs.io/en/latest/parameters.html

    Args:
        :k (Union[float, np.ndarray]): Wavenumber in unit h Mpc-1
        :z (Union[float, np.ndarray]): Redshift
        :Omega_m (Union[float, np.ndarray]): Density of matter.
        :sigma_8 (Union[float, np.ndarray]): amplitude of matter fluctuations
        :A_SN1 (Union[float, np.ndarray]): Thermal energy injected in each SNII event
        :A_AGN1 (Union[float, np.ndarray]): Scaling of the black hole Bondi accretion rate
        :A_SN2 (Union[float, np.ndarray]): Metallicity dependence of the stellar feedback fraction per unit stellar mass
        :A_AGN2 (Union[float, np.ndarray]): Temperature jump of gas particles in AGN feedback events

    Returns:
        :S (Union[float, np.ndarray)): Baryonic suppression of the matter power spectrum
    """

    alpha_1 = 0.272*Omega_m
    alpha_2 = 0.22
    alpha_3 = 0.0168
    alpha_4 = 0.074
    alpha_5 = 0.0204/Omega_m**1.625
    alpha_6 = 0.004/Omega_m**1.625
    alpha_7 = 1.1166
    alpha_8 = 54.3014*Omega_m**1.625
    alpha_9 = 0.12055*Omega_m**1.625*(4822.2*A_SN1 + 8228.9*Omega_m)
    alpha_10 = (-0.2588*A_AGN1 + 6.13845*A_SN1) / \
        (Omega_m**1.625*(6.528*A_AGN2 + 12.855*A_SN1))
    alpha_11 = (0.9788 - 0.1659*A_SN1)*(-0.3745*A_AGN1 + 8.877 *
                                        A_SN1)/(Omega_m**1.625*(6.528*A_AGN2 + 12.855*A_SN1))
    alpha_12 = 9.2437
    alpha_13 = 9.2965*A_SN2 + 18.4
    alpha_14 = (0.034*Omega_m - 0.02*sigma_8)/Omega_m**1.625
    alpha_15 = 2.287
    alpha_16 = 130.0
    alpha_17 = 0.565
    log_s = alpha_1**(alpha_2/k + alpha_3*k + alpha_4*z)*(alpha_14 - alpha_5*k + alpha_6*z + (alpha_10*k + alpha_11)/(alpha_12*z +
                                                                                                                      alpha_13 + k) + (alpha_7*k*z + k**2)/(alpha_8*k + alpha_9)) + alpha_15*torch.exp(-alpha_16*torch.exp(-z)/k - torch.exp(alpha_17*z))
    s = torch.exp(log_s)

    return s


def syren_baryon_Baryonification(k, z, sigma8, Om, Ob, logMc, logeta, logbeta, logM1, logMinn, logthetainn):
    """"
    Compute nonlinear P(k) for the cosmology of interest

    Args:
        :k (Union[float, np.ndarray]): k values to evaluate P(k) at [h / Mpc]
        :z (Union[float, np.ndarray]): Redshift.
        :sigma8 (Union[float, np.ndarray]): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :Om (Union[float, np.ndarray]): The z=0 total matter density parameter, Omega_m
        :Ob (Union[float, np.ndarray]): The z=0 baryonic density parameter, Omega_b
        :logMc (Union[float, np.ndarray]): (Log of) mass fraction of hot gas in haloes
        :logeta (Union[float, np.ndarray]): (Log of) extent of ejected gas
        :logbeta (Union[float, np.ndarray]): (Log of) mass fraction of hot gas in haloes
        :logM1 (Union[float, np.ndarray]): (Log of) characteristic halo mass scale for central galaxy
        :logMinn (Union[float, np.ndarray]): (Log of) property describing density profile of hot gas in haloes
        :logthetainn (Union[float, np.ndarray]): (Log of) property describing density profile of hot gas in haloes

    Returns:
        :S (np.ndarray): Ratio of baryonic to non-baryonic power spectrum at ks provided.

    """

    alpha_1 = 2.4663*(5.823*Ob)**(3.23*Om)
    alpha_2 = 5.823*Ob
    alpha_3 = 0.4355
    alpha_4 = 0.0495*logM1
    alpha_5 = 0.2479
    alpha_6 = -0.27*logeta
    alpha_7 = 0.0015*logMinn + 0.0176*logthetainn - \
        0.0015*(3.9144*sigma8)**(0.2058*logM1)
    alpha_8 = 0.016*logMc
    alpha_9 = 0.0706*logMc
    alpha_10 = 0.0383
    alpha_11 = -6.5*logbeta - 6.5
    log_s = alpha_1*alpha_2**(alpha_3*z)*k*(alpha_4 - (alpha_5*k)**alpha_6) * \
        (alpha_7*k + alpha_8**((alpha_10*z + alpha_9)**alpha_11))
    s = torch.exp(log_s)

    return s


def epsilon_Astrid(k, z):
    """
    Calculate the typical error of the prediction of `Astrid`.

    Args:
        :k (Union[float, np.ndarray]): k values to evaluate P(k) at [h / Mpc]
        :z (Union[float, np.ndarray]): Redshift.

    Returns:
        :epsilon (Union[float, np.ndarray]): Covariance of the prediction and the actual S of Astrid.
    """

    alpha_1 = 0.0202
    alpha_2 = 0.18327
    alpha_3 = 1.3
    alpha_4 = 0.25285
    epsilon = alpha_1*k / \
        (alpha_2*k*torch.exp(-alpha_3*z) + torch.exp(alpha_4*z))

    return epsilon


def epsilon_IllustrisTNG(k, z):
    """
    Calculate the typical error of the prediction of `IllustrisTNG`.

    Args:
        :k (Union[float, np.ndarray]): k values to evaluate P(k) at [h / Mpc]
        :z (Union[float, np.ndarray]): Redshift.

    Returns:
        :epsilon (Union[float, np.ndarray]): Covariance of the prediction and the actual S of IllustrisTNG.
    """

    alpha_1 = 17.119
    alpha_2 = 0.63
    alpha_3 = 48.797
    alpha_4 = 19.238
    epsilon = k/(alpha_1*k*torch.exp(-alpha_2*z) + alpha_3*z + alpha_4)

    return epsilon


def epsilon_SIMBA(k, z):
    """
    Calculate the typical error of the prediction of `SIMBA`.

    Args:
        :k (Union[float, np.ndarray]): k values to evaluate P(k) at [h / Mpc]
        :z (Union[float, np.ndarray]): Redshift.

    Returns:
        :epsilon (Union[float, np.ndarray]): Covariance of the prediction and the actual S of SIMBA.
    """

    alpha_1 = 0.05904
    alpha_2 = 0.43
    alpha_3 = 0.63239
    alpha_4 = 0.02852
    epsilon = alpha_1*torch.exp(-alpha_2*torch.exp(alpha_3*z)/k + alpha_4*k)

    return epsilon


def epsilon_Swift_EAGLE(k, z):
    """
    Calculate the typical error of the prediction of `Swift_EAGLE`.

    Args:
        :k (Union[float, np.ndarray]): k values to evaluate P(k) at [h / Mpc]
        :z (Union[float, np.ndarray]): Redshift.

    Returns:
        :epsilon (Union[float, np.ndarray]): Covariance of the prediction and the actual S of Swift-EAGLE.
    """

    alpha_1 = 0.032
    alpha_2 = 0.54
    alpha_3 = 0.363
    epsilon = alpha_1*k/(alpha_2*k + torch.exp(alpha_3*z))

    return epsilon
