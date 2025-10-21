import numpy as np

def symbolic_2f1_comoving(x):
    
    # Operon fit but ensure 2F1(2/3,1,7/6;0) = 1
    # c = [0.920695062149842, 0.986171932633297, 1.424994, 0.918750289187582, 0.465164]
    c = [0.9207, 0.98617, 1.42499, 0.91875, 0.46516]
    f = ((c[0] + c[1]**c[2]) / (c[0] + (c[1] - x)**c[2] - c[3]*x)) ** c[4]
    
    return f

def symbolic_radial_comoving(Omm, z):
    """
    Compute the radial comoving distance for a LCDM universe at a given
    redshift using a symbolic approximations. This is defined as
    
    chi(a) = R_H \int_a^1 dx / (x^2 sqrt(Omm x^{-3} + 1 - Omm))
    
    The integral is proportional to 2F1(2/3,1,7/6; (Omm-1)/Omm x^3)
    so we use the symbolic approximation for this

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :z (float): Redshift to evaluate at

    Returns:
        :chi (float): The comoving distance at redshift z [h^{-1} Mpc]

    """
    
    # Lower limit
    a = 1/(1+z)
    x = (Omm - 1) / Omm * a ** 3
    lower = a**2 * np.sqrt(Omm * a**(-3) + 1 - Omm) * symbolic_2f1_comoving(x)
    
    # Upper limit
    x = (Omm - 1) / Omm
    upper = symbolic_2f1_comoving(x)
    
    # Hubble radius in [h^{-1} Mpc]
    rh = 2997.92458  # h^{-1} Mpc
    
    chi = 2 * rh / Omm * (upper - lower)
    
    return chi
    
    