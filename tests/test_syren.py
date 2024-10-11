import numpy as np
import math
import camb
import unittest
import torch
import itertools
import scipy.special

import symbolic_pofk.linear as linear
import symbolic_pofk.syrenhalofit as syrenhalofit

import symbolic_pofk.linear_plus as linear_plus
import symbolic_pofk.syren_plus as syren_plus

import symbolic_pofk.pytorch.linear as torch_linear
import symbolic_pofk.pytorch.syrenhalofit as torch_syrenhalofit
import symbolic_pofk.pytorch.linear_plus as torch_linear_plus
import symbolic_pofk.pytorch.syren_plus as torch_syren_plus
import symbolic_pofk.pytorch.utils as torch_utils


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
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=0.0,
                       omk=0,
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
    pk_camb = pk_camb[0, :]
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
    pk_with = linear.pk_EisensteinHu_zb(
        k, sigma8, Om, Ob, h, ns, use_colossus=False)
    pk_without = linear.pk_EisensteinHu_zb(
        k, sigma8, Om, Ob, h, ns, use_colossus=True)
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
    assert np.allclose(np.log(pk_camb_halofit),
                       np.log(pk_takahashi), atol=1e-2)

    # Check halofit and syren give similar results
    pk_syrenhalofit = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a,
                                               emulator='fiducial', extrapolate=True, which_params='Bartlett',
                                               add_correction=True)
    assert np.allclose(np.log(pk_bartlett), np.log(pk_syrenhalofit), atol=1e-1)

    # Check halofit correction is what is expected
    A_emu = syrenhalofit.A_emulated(k, sigma8, Om, Ob, h, ns, a)
    A_check = pk_syrenhalofit / pk_bartlett - 1
    assert np.allclose(A_emu, A_check, atol=1e-4)


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

    # Redshift
    z = 1
    a = 1 / (1+z)

    theta = torch.tensor([As, Om, Ob, h, ns, a],
                         requires_grad=True).reshape(1, -1)

    # Old sigma8 conversion
    sigma8_old = linear.As_to_sigma8(As, Om, Ob, h, ns, old_equation=True)
    torch_sigma8_old = torch_linear.As_to_sigma8(
        theta[:, :5], old_equation=True)
    assert math.isclose(sigma8_old, torch_sigma8_old.item(), rel_tol=1e-5)

    # New sigma8 conversion
    sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
    torch_sigma8 = torch_linear.As_to_sigma8(theta[:, :5])
    assert math.isclose(sigma8, torch_sigma8.item(), rel_tol=1e-5)

    theta_sig8 = theta.clone().detach()

    # Old inverse sigma8 conversion
    theta_sig8[:, 0] = torch.tensor([sigma8_old])
    As_new = linear.sigma8_to_As(sigma8_old, Om, Ob, h, ns, old_equation=True)
    torch_As_new = torch_linear.sigma8_to_As(
        theta_sig8[:, :5], old_equation=True)
    assert math.isclose(As_new, torch_As_new.item(), rel_tol=1e-5)

    # New inverse sigma8 conversion
    theta_sig8[:, 0] = torch.tensor([sigma8])
    As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns)
    torch_As_new = torch_linear.sigma8_to_As(theta_sig8[:, :5])
    assert math.isclose(As_new, torch_As_new.item(), rel_tol=1e-5)

    # Eisenstein and Hu
    for integral_norm in [True, False]:
        pk_eh = linear.pk_EisensteinHu_zb(
            k, sigma8, Om, Ob, h, ns, integral_norm=integral_norm, use_colossus=False)
        torch_pk_eh = torch_linear.pk_EisensteinHu_zb(
            kt, theta_sig8[:, :5], integral_norm=integral_norm)
        assert np.allclose(pk_eh, torch_pk_eh.detach().numpy(), rtol=1e-5)

    # Check syren linear at various z matches numpy version
    for emulator in ['fiducial', 'max_precision']:
        for extrapolate in [True, False]:
            for use_approx_D in [True, False]:
                for z_ in np.linspace(0, 2, 5):
                    a_ = 1 / (1+z_)
                    theta_temp = torch.tensor([sigma8, Om, Ob, h, ns, a_],
                                            requires_grad=True).reshape(1, -1)
                    pk = linear.plin_emulated(
                        k, sigma8, Om, Ob, h, ns, a_, emulator=emulator, extrapolate=extrapolate)
                    torch_pk = torch_linear.plin_emulated(
                        kt, theta_temp, emulator=emulator, extrapolate=extrapolate, use_approx_D=use_approx_D)
                    assert np.allclose(pk, torch_pk.detach().numpy(), rtol=5e-3)

    # Check asking for a different emulator raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        torch_linear.plin_emulated,
        kt, theta_sig8,
        emulator='something_else',
    )

    # Check halofit at various z matches numpy version
    for z_ in np.linspace(0, 2, 5):
        a_ = 1 / (1+z_)
        theta_temp = torch.tensor([sigma8, Om, Ob, h, ns, a_],
                                  requires_grad=True).reshape(1, -1)

        #  Halofit variables
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
        torch_A = torch_syrenhalofit.A_emulated(
            kt, theta_temp, ksigma=ksigma, neff=neff, C=C)
        assert np.allclose(A, torch_A.detach().numpy(), atol=1e-5)

        # Now all combinations of halofit
        combinations = list(itertools.product(['fiducial', 'max_precision'], [
                            'Bartlett', 'Takahashi'], [True, False], [True, False]))
        for emulator, which_params, add_correction, extrapolate in combinations:
            pk = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a_,
                                          emulator=emulator, which_params=which_params, add_correction=add_correction, extrapolate=extrapolate)
            torch_pk = torch_syrenhalofit.run_halofit(
                kt, theta_temp, emulator=emulator, which_params=which_params, add_correction=add_correction, extrapolate=extrapolate)[0]
            assert np.allclose(pk, torch_pk.detach().numpy(), rtol=5e-3)

    # Check asking for a different halofit raises NotImplementedError
    unittest.TestCase().assertRaises(
        NotImplementedError,
        torch_syrenhalofit.run_halofit,
        kt, theta_sig8,
        emulator='fiducial',
        which_params='something_else',
    )

    return


def test_utils_torch():

    #  Test Simpon's rule integrator
    test_cases = [
        # Valid cases
        {
            'y': torch.tensor([0, 1, 4], dtype=torch.float64),
            'x': torch.tensor([0, 1, 2], dtype=torch.float64),
            'dx': None,
            'expected': (8 / 3.0),  # Integral of y = x^2 from 0 to 2
            'axis': -1
        },
        {
            'y': torch.tensor([0, 1, 4, 9, 16], dtype=torch.float64),
            'x': torch.tensor([0, 1, 2, 3, 4], dtype=torch.float64),
            'dx': None,
            'expected': (64 / 15.0),  # Integral of y = x^2 from 0 to 4
            'axis': -1
        },
        {
            'y': torch.tensor([0, 1, 4, 9, 16], dtype=torch.float64),
            'x': None,
            'dx': 1.0,
            'expected': (64 / 15.0),  # Integral of y = x^2 from 0 to 4
            'axis': -1
        },
        {
            'y': torch.tensor([0, 1, 4, 9], dtype=torch.float64),
            'x': None,
            'dx': 1.0,
            'expected': (9),  # Integral of y = x^2 from 0 to 3
            'axis': -1
        },
        {
            'y': torch.tensor([1, 2, 1], dtype=torch.float64),
            'x': torch.tensor([0, 1, 2], dtype=torch.float64),
            'dx': None,
            'expected': 1.0,  # Integral of y = 1 - |x-1| from 0 to 2
            'axis': -1
        },

        # Edge cases
        {
            'y': torch.tensor([]),
            'x': torch.tensor([]),
            'dx': 1.0,
            'expected': ValueError,
            'axis': -1
        },
        {
            'y': torch.tensor([1]),
            'x': torch.tensor([0]),
            'dx': 1.0,
            'expected': ValueError,
            'axis': -1
        },
        {
            'y': torch.tensor([1, 2]),
            'x': torch.tensor([0, 1]),
            'dx': None,
            'expected': 1.0,  # Trapezoidal result for two points
            'axis': -1
        },
        {
            'y': torch.tensor([1, 2, 3]),
            'x': torch.tensor([0, 1, 2]),
            'dx': None,
            'expected': 2.0,  # Valid case with three points
            'axis': -1
        },
        {
            'y': torch.sin(torch.linspace(0, np.pi, 5)),
            'x': torch.linspace(0, np.pi, 5),
            'dx': None,
            'expected': 2.0,  # Integral of sin(x) from 0 to π
            'axis': -1
        },
        {
            'y': torch.exp(torch.linspace(0, 1, 5)),
            'x': torch.linspace(0, 1, 5),
            'dx': None,
            'expected': (np.exp(1) - 1),  # Integral of exp(x) from 0 to 1
            'axis': -1
        },
        {
            'y': torch.Tensor([1, 2, 3]),
            'x': torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            'dx': None,
            'expected': ValueError,  # y is 1-d by x is multi-d
            'axis': -1
        },
        {
            'y': torch.exp(torch.linspace(0, 1, 5)),
            'x': torch.linspace(0, 1, 6),
            'dx': None,
            'expected': ValueError,  # x and y have different lengths
            'axis': -1
        },
    ]

    for case in test_cases:

        y = case['y']
        x = case['x']
        dx = case['dx']
        expected = case['expected']
        axis = case['axis']

        if isinstance(expected, type) and issubclass(expected, Exception):
            try:
                result = torch_utils.simpson(y, x=x, dx=dx, axis=axis)
                raise Exception(f"Expected exception for inputs {y}, {x}, {dx}, {axis}")
            except expected:
                pass  # Correctly raised expected exception
            except Exception:
                raise Exception(f"Unexpected exception for inputs {y}, {x}, {dx}, {axis}")
        else:
            # Run the custom Simpson's rule
            result = torch_utils.simpson(y, x=x, dx=dx, axis=axis)
            # Compare with the SciPy Simpson's rule
            if x is None:
                scipy_result = scipy.integrate.simpson(y.numpy(), dx=dx)
            else:
                scipy_result = scipy.integrate.simpson(y.numpy(), x=x.numpy())

            # Check if results are similar within a tolerance
            np.testing.assert_allclose(result.numpy(
            ), scipy_result, rtol=1e-5, err_msg=f"Failed for inputs {y}, {x}, {dx}, {axis}")

    # Check _basic_simpson behaves as expected when stary is None (it should be set to 0)
    y = torch.tensor([0, 1, 4], dtype=torch.float64)
    x = torch.tensor([0, 1, 2], dtype=torch.float64)
    dx = None
    axis = -1
    result_0 = torch_utils._basic_simpson(y, 0, 2, x, dx, axis)
    result_None = torch_utils._basic_simpson(y, None, 2, x, dx, axis)
    np.testing.assert_equal(result_0, result_None)

    #  Test hypergeometric function
    test_cases = [
        # (a, b, c, z, expected_result_function, tolerance)
        (0.5, 0.5, 1.0, torch.tensor([0.5])),
        (0.5, 0.5, 1.0, torch.tensor([1.0])),
        (0.5, 0.5, 1.0, torch.tensor([2.0])),
        (0.0, 0.0, 1.0, torch.tensor([0.5])),
        (1000.0, 1000.0, 2000.0, torch.tensor([0.5])),
        (0.5, 0.5, 1.0, torch.tensor([-0.5])),
    ]
    for i, (a, b, c, z) in enumerate(test_cases):
        expected = scipy.special.hyp2f1(a, b, c, z).item()
        result = torch_utils.hyp2f1(a, b, c, z)
        assert np.isclose(result, torch.tensor(expected), atol=1e-6), \
            f"Test case {i+1} failed: Expected {expected}, got {result.item()}"

    # Test error raised for hypergeometric series for invalid argument
    unittest.TestCase().assertRaises(
        ValueError,
        torch_utils.hypergeometric_series,
        0.5, 0.5, 1.0, torch.tensor([2.0]),
        max_iter=10000,
        tolerance=1.0e-6
    )

    return
  
  
def test_syren_plus():

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
    mnu = 0.10
    w0 = -0.9
    wa = 0.1

    # Redshift
    z = 1
    a = 1 / (1+z)
    
    # Get sigma8 for this As
    sigma8 = linear_plus.As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa)

    # See what As you get in reverse
    As_new = linear_plus.sigma8_to_As(sigma8, Om, Ob, h, ns, mnu, w0, wa)
    assert math.isclose(As, As_new, rel_tol=1e-2)

    # Compute P(k) using camb
    num_massive_neutrinos = (3
                             if mnu != 0.0
                             else 0)
    Oc = Om - Ob - mnu / 93.14 / h ** 2
    redshift = 1 / a - 1
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100 * h,
        ombh2=(Ob * h ** 2),
        omch2=(Oc * h**2),
        omk=0,
        neutrino_hierarchy='degenerate',
        num_massive_neutrinos=num_massive_neutrinos,
        mnu=mnu,
        standard_neutrino_neff=3.046,
        tau=tau
    )
    pars.set_dark_energy(w=w0, wa=wa)
    pars.InitPower.set_params(ns=ns, As=As*1e-9, r=0)
    pars.set_matter_power(redshifts=[redshift], kmax=k[-1],
                          accurate_massive_neutrino_transfers=True)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    index = 6
    _, _, plin_camb = results.get_matter_power_spectrum(
        var1=(1 + index),
        var2=(1 + index),
        minkh=k.min(),
        maxkh=k.max(),
        npoints=len(k)
    )
    plin_camb = plin_camb[0]

    # Get emulated power spectrum
    plin_syren_plus = linear_plus.plin_plus_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a=a)

    # Check that the linear emulator is close to camb
    assert np.allclose(np.log(plin_camb), np.log(plin_syren_plus), atol=1e-2)

    # Get camb halofit
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    results = camb.get_results(pars)
    nk = len(k)
    kh, z, pk_camb_halofit = results.get_matter_power_spectrum(
        minkh=k[0], maxkh=k[-1], npoints=nk)
    pk_camb_halofit = pk_camb_halofit[0]

    # Get syren halofit
    pk_syren_plus = syren_plus.pnl_plus_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a)

    # Check that the emulator is close to camb
    assert np.allclose(np.log(pk_camb_halofit),
                       np.log(pk_syren_plus), atol=1e-1)

    #  Check that the result at minimum k is close to linear
    assert np.allclose(np.log(plin_syren_plus[0]), np.log(
        pk_syren_plus[0]), atol=1e-2)

    return


def test_torch_syren_plus():

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
    mnu = 0.10
    w0 = -0.9
    wa = 0.1

    # Redshift
    z = 1
    a = 1 / (1+z)

    # Get numpy versions
    plin_numpy = linear_plus.plin_plus_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a=a)
    pnl_numpy = syren_plus.pnl_plus_emulated(
        k, As, Om, Ob, h, ns, mnu, w0, wa, a)

    # Get torch versions
    theta = torch.tensor([As, Om, Ob, h, ns, mnu, w0, wa, a],
                         requires_grad=True).reshape(1, -1)
    k = torch.tensor(k, requires_grad=True)
    plin_torch = torch_linear_plus.plin_plus_emulated(k, theta)
    pnl_torch = torch_syren_plus.pnl_plus_emulated(k, theta)

    # Check that the results are close
    assert np.allclose(np.log(plin_numpy), np.log(
        plin_torch.detach().numpy()), atol=1e-6)
    assert np.allclose(np.log(pnl_numpy), np.log(
        pnl_torch.detach().numpy()), atol=1e-6)

    return
