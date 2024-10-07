import torch

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def logF_fiducial(k_batch, theta_batch):
    """
    Compute the emulated logarithm of the ratio between the true linear power spectrum 
    and the Eisenstein & Hu 1998 fit for LCDM as given in Bartlett et al. 2023.

    Args:
        :k_batch (torch.Tensor): tensor containing the k values to evaluate P(k) at [h / Mpc] with shape (n_k,1)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 5),
            the 9 parameters are :
                :sigma8: Root-mean-square density fluctuation when the linearly
                    evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
                Om: The z=0 total matter density parameter
                Ob: The z=0 baryonic density parameter
                h: Hubble constant, H0, divided by 100 km/s/Mpc
                ns: Spectral tilt of primordial power spectrum

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
    
    _, Om, Ob, h, _ = theta_batch.unbind(dim=1)
        
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
