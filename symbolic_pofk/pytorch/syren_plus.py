import torch
from .linear_plus import plin_plus_emulated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pnl_plus_emulated(k, theta_batch):

    '''
    Compute the non-linear power spectrum using a symbolic approximation
    to the linear power spectrum. 
    
    Args:
        :k (torch.Tensor): k values [h/Mpc] with shape (n_k)
        :theta_batch (torch.Tensor): tensor containing the parameters with shape (batch_size, 9)
        
    Returns:
        :Pk (torch.Tensor): computed nonlinear power spectrum for each k and theta in the batch
    '''
    
    g= torch.tensor([0.2107, 0.0035, 0.0667, 0.0442, 1.2809, 0.2287, 0.1122, 4.3318, 1.1857, 3.3117, 14.2829, 
                 0.9039, 0.0749, 0.0741, 0.1277, 27.6818, 24.8736, 0.6264, 0.3035, 0.6069, 0.7882, 0.4811, 
                 1.4326, 1.8971, 0.0271, 0.9635, 0.0264, 22.9213, 71.1658, 0.0371, 0.0099, 210.3925, 0.2555], device=device)
    
    #calculate the linear power spectrum using the emulated model
    P_lin = torch.log10(plin_plus_emulated(k, theta_batch).T)

    k_batch = k.unsqueeze(1).to(device) 

    As, Om, Ob, h, ns, mnu, w0, wa, a = theta_batch.unbind(dim=1)


    term1 = P_lin
   

    numerator1 = g[0] * k_batch * (g[1] * k_batch)**(g[2] * Om - g[3] * As)
    denominator1_part1 = (g[4] * k_batch**(-g[5]) - g[6] * P_lin)**(g[7] * P_lin + g[8] * wa + g[9] * w0 - g[10])
    denominator1_part2 = (g[11] * k_batch**g[12] + g[13] * P_lin - g[14] * Om)**(g[15] * a - g[16] * ns)
    term2 = numerator1 / (denominator1_part1 + denominator1_part2)

    numerator2 = (g[17] * a - g[18] * P_lin + g[19]) * k_batch
    denominator2 = (g[20] * Om + g[21] * k_batch + g[22] * ns - g[23] +
                    (g[24] * P_lin + g[25] * k_batch**g[26])**(g[27] * a - g[28] * ns))
    term3 = numerator2 / denominator2

    term4 = g[29] * k_batch

    term5 = (g[30] * k_batch)**((g[31] * k_batch)**(-a * g[32]))

    # Combine all terms
    result = term1 + term2 + term3 - term4 - term5

    bias = pnl_bias(k_batch)
    result = result - bias

    return torch.pow(10,result).T


def pnl_bias(k_batch):
    """
    the offset of the emulated non-linear power spectrum from euclidemulator2
    Parameters:
    Args:
        :k_batch (torch.Tensor): k values [h/Mpc] with shape (n_k,1)
        
    Returns:
    - offset: The offset of the emulated non-linear log_10 P(k)
    """
    

    h=torch.tensor([  0.5787,   2.3485,  27.3829,  16.4236,  97.3766,  90.9764,
                11.2046,  2447.2 , 11376.93], device=device)

    term1 = ((h[1] * k_batch) - torch.cos(h[3]*torch.cos(h[2] * k_batch)))*torch.cos(h[4] * k_batch)
    denominator = -h[7]*torch.log(h[6] * k_batch)  + (h[8] * k_batch)

    offset = ((h[0] + term1 + torch.cos(h[5] * k_batch))) / denominator
    
    return offset