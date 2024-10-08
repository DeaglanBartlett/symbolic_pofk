import numpy as np
import math
import camb
import unittest
import torch
import itertools

import symbolic_pofk.linear as linear
import symbolic_pofk.syrenhalofit as syrenhalofit

import symbolic_pofk.pytorch.linear as torch_linear
import symbolic_pofk.pytorch.syrenhalofit as torch_syrenhalofit

def test_lcdm():

    # Define k range
    kmin = 9e-3
    kmax = 9
    nk = 400
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # Cosmological parameters
    As = 2.105  # 10^9 A_s
    h = 0.6766
    Om = 0.3111
    Ob = 0.02242 / h ** 2
    ns = 0.9665
    tau = 0.0561
    mnu = 0.0
    
    # Redshift
    z = 1
    a = 1 / (1+z)
    
    # Get sigma8 for this As
    sigma8_old = linear.As_to_sigma8(As, Om, Ob, h, ns, old_equation=True)
    sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
    assert math.isclose(sigma8_old, sigma8, rel_tol=1e-2)

    # See what As you get in reverse
    As_new = linear.sigma8_to_As(sigma8_old, Om, Ob, h, ns, old_equation=True)
    assert math.isclose(As, As_new, rel_tol=1e-2)
    As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns)
    assert math.isclose(As, As_new, rel_tol=1e-2)
    
    # Compute P(k) using camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = h*100,
                       ombh2 = Ob * h ** 2,
                       omch2 = (Om - Ob) * h ** 2,
                       mnu = 0.0,
                       omk = 0,
                       tau=tau,)
    As_fid = 2.0e-9
    pars.InitPower.set_params(As=As_fid, ns=ns, r=0)
    pars.set_matter_power(redshifts=[0.], kmax=k[-1])
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb = results.get_sigma8()[0]
    As_new = (sigma8 / sigma8_camb) ** 2 * As_fid
    print('As from camb', As_new)
    pars.InitPower.set_params(As=As_new, ns=ns, r=0)
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
                            minkh=k.min(), maxkh=k.max(), npoints=len(k))
    pk_camb = pk_camb[0,:]
    pk_eh = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)
    logF_camb = np.log(pk_camb / pk_eh)
    
    # Check two linear emulators give correct results and are close to camb value
    for emulator in ['fiducial', 'max_precision']:
        pk = linear.plin_emulated(k, sigma8, Om, Ob, h, ns,
            emulator=emulator, extrapolate=True)
        assert len(pk) == len(k)
        assert isinstance(pk, np.ndarray)
        logF = np.log(pk / pk_eh)
        assert np.allclose(logF_camb, logF, atol=1e-2)
        
        # Check warning raised if run outside range of fit
        knew = np.logspace(np.log10(kmin/10), np.log10(kmax*10), nk)
        with unittest.TestCase().assertWarns(UserWarning) as cm:
            pk = linear.plin_emulated(knew, sigma8, Om, Ob, h, ns,
                emulator=emulator, extrapolate=False)
        # Verify the warning message
        unittest.TestCase().assertEqual(str(cm.warning),
            "Not using Bartlett et al. formula outside tested regime")
            
    # Check asking for a different emulator raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        linear.plin_emulated,
        k, sigma8, Om, Ob, h, ns,
        emulator='something_else',
        extrapolate=True
    )
        
    # Check that there is little difference between colossus and not using it
    pk_with = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, use_colossus=False)
    pk_without = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, use_colossus=True)
    np.allclose(np.log(pk_with), np.log(pk_without), atol=1e-2)
        
    # Check halofit give similar results
    pk_bartlett = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a,
        emulator='fiducial', extrapolate=True, which_params='Bartlett',
        add_correction=False)
    pk_takahashi = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a,
        emulator='fiducial', extrapolate=True, which_params='Takahashi',
        add_correction=False)
    assert np.allclose(np.log(pk_bartlett), np.log(pk_takahashi), atol=1e-1)
    
    # Check asking for a different halofit raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        syrenhalofit.run_halofit,
        k, sigma8, Om, Ob, h, ns, a,
        emulator='fiducial',
        extrapolate=True,
        which_params='something_else',
    )

    # Check CAMB halofit similar to Takahashi above
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=mnu,
                       omk=0,
                       tau=tau,)
    pars.InitPower.set_params(As=As*1.e-9, ns=ns, r=0)
    redshift = 1 / a - 1
    pars.set_matter_power(redshifts=[redshift], kmax=k[-1])
    results = camb.get_results(pars)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    results = camb.get_results(pars)
    nk = len(k)
    kh, z, pk_camb_halofit = results.get_matter_power_spectrum(
        minkh=k[0], maxkh=k[-1], npoints=nk)
    pk_camb_halofit = pk_camb_halofit[0]
    assert np.allclose(np.log(pk_camb_halofit), np.log(pk_takahashi), atol=1e-2)

    # Check halofit and syren give similar results
    pk_syrenhalofit = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a,
        emulator='fiducial', extrapolate=True, which_params='Bartlett',
        add_correction=True)
    assert np.allclose(np.log(pk_bartlett), np.log(pk_syrenhalofit), atol=1e-1)
    
    # Check halofit correction is what is expected
    A_emu = syrenhalofit.A_emulated(k, sigma8, Om, Ob, h, ns, a)
    A_check = pk_syrenhalofit / pk_bartlett - 1
    assert np.allclose(A_emu, A_check, atol=1e-4)
        
    return


def test_lcdm_torch():

    # Define k range
    kmin = 9e-3
    kmax = 9
    nk = 400
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    kt = torch.tensor(k, requires_grad=True)

    # Cosmological parameters
    As = 2.105  # 10^9 A_s
    h = 0.6766
    Om = 0.3111
    Ob = 0.02242 / h ** 2
    ns = 0.9665
    tau = 0.0561

    # Redshift
    z = 1
    a = 1 / (1+z)
    
    theta = torch.tensor([As, Om, Ob, h, ns, a],
                         requires_grad=True).reshape(1, -1)
    
    # Old sigma8 conversion
    sigma8_old = linear.As_to_sigma8(As, Om, Ob, h, ns, old_equation=True)
    torch_sigma8_old = torch_linear.As_to_sigma8(theta[:,:5], old_equation=True)
    assert math.isclose(sigma8_old, torch_sigma8_old.item(), rel_tol=1e-5)

    # New sigma8 conversion
    sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
    torch_sigma8 = torch_linear.As_to_sigma8(theta[:,:5])
    assert math.isclose(sigma8, torch_sigma8.item(), rel_tol=1e-5)

    theta_sig8 = theta.clone().detach()

    # Old inverse sigma8 conversion
    theta_sig8[:,0] = torch.tensor([sigma8_old])
    As_new = linear.sigma8_to_As(sigma8_old, Om, Ob, h, ns, old_equation=True)
    torch_As_new = torch_linear.sigma8_to_As(theta_sig8[:,:5], old_equation=True)
    assert math.isclose(As_new, torch_As_new.item(), rel_tol=1e-5)

    # New inverse sigma8 conversion
    theta_sig8[:,0] = torch.tensor([sigma8])
    As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns)
    torch_As_new = torch_linear.sigma8_to_As(theta_sig8[:,:5])
    assert math.isclose(As_new, torch_As_new.item(), rel_tol=1e-5)

    # Eisenstein and Hu
    for integral_norm in [True, False]:
        pk_eh = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns, integral_norm=integral_norm, use_colossus=False)
        torch_pk_eh = torch_linear.pk_EisensteinHu_zb(kt, theta_sig8[:,:5], integral_norm=integral_norm)
        assert np.allclose(pk_eh, torch_pk_eh.detach().numpy(), rtol=1e-5)

    # Check syren linear at various z matches numpy version
    for emulator in ['fiducial', 'max_precision']:
        for z_ in np.linspace(0, 2, 5):
            a_ = 1 / (1+z_)
            theta_temp = torch.tensor([sigma8, Om, Ob, h, ns, a_],
                         requires_grad=True).reshape(1, -1)
            pk = linear.plin_emulated(k, sigma8, Om, Ob, h, ns, a_, emulator=emulator, extrapolate=True)
            torch_pk = torch_linear.plin_emulated(kt, theta_temp, emulator=emulator)
            assert np.allclose(pk, torch_pk.detach().numpy(), rtol=1e-4)

    # Check asking for a different emulator raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        torch_linear.plin_emulated,
        kt, theta_sig8,
        emulator='something_else',
    )

    # Check halofit at various z matches numpy version
    for z_ in  np.linspace(0, 2, 5):
        a_ = 1 / (1+z_)
        theta_temp = torch.tensor([sigma8, Om, Ob, h, ns, a_],
                        requires_grad=True).reshape(1, -1)
        
        # Halofit variables
        ksigma = syrenhalofit.ksigma_emulated(sigma8, Om, Ob, h, ns, a_)
        torch_ksigma = torch_syrenhalofit.ksigma_emulated(theta_temp)
        assert np.allclose(ksigma, torch_ksigma.detach().numpy(), rtol=1e-5)
        neff = syrenhalofit.neff_emulated(sigma8, Om, Ob, h, ns, a_)
        torch_neff = torch_syrenhalofit.neff_emulated(theta_temp)
        assert np.allclose(neff, torch_neff.detach().numpy(), rtol=1e-5)
        C = syrenhalofit.C_emulated(sigma8, Om, Ob, h, ns, a_)
        torch_C = torch_syrenhalofit.C_emulated(theta_temp)
        assert np.allclose(C, torch_C.detach().numpy(), rtol=1e-5)

        # Correction to Halofit
        A = syrenhalofit.A_emulated(k, sigma8, Om, Ob, h, ns, a_)
        torch_A = torch_syrenhalofit.A_emulated(kt, theta_temp)
        assert np.allclose(A, torch_A.detach().numpy(), atol=1e-5)
        torch_A = torch_syrenhalofit.A_emulated(kt, theta_temp, ksigma=ksigma, neff=neff, C=C)
        assert np.allclose(A, torch_A.detach().numpy(), atol=1e-5)

        # Now all combinations of halofit
        combinations = list(itertools.product(['fiducial', 'max_precision'], ['Bartlett', 'Takahashi'], [True, False]))
        for emulator, which_params, add_correction in combinations:
            pk = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a_,
                emulator=emulator, which_params=which_params, add_correction=add_correction)
            torch_pk = torch_syrenhalofit.run_halofit(kt, theta_temp, emulator=emulator, which_params=which_params, add_correction=add_correction)
            assert np.allclose(pk, torch_pk.detach().numpy(), rtol=1e-4)

    # Check asking for a different halofit raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        torch_syrenhalofit.run_halofit,
        kt, theta_sig8,
        emulator='fiducial',
        which_params='something_else',
    )

    return
