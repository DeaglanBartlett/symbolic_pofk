import numpy as np
import scipy.special
import camb

import symbolic_pofk.wider_syren.background as background
import symbolic_pofk.wider_syren.linear as linear
import symbolic_pofk.wider_syren.halofit as halofit

def test_comoving_hyper():
    
    Om_min, Om_max = 0.1, 0.5
    z_min, z_max = 0, 10
    a_min = 1 / (1 + z_max)
    a_max = 1 / (1 + z_min)
    x_min = (1 - 1 / Om_min) * a_max ** 3
    x_max = (1 - 1 / Om_max) * a_min ** 3

    samples = np.linspace(x_min, x_max)

    true = scipy.special.hyp2f1(2/3, 1, 7/6, samples)
    pred = background.symbolic_2f1_comoving(samples)

    assert np.allclose(pred, true, rtol=1e-3), 'symbolic_2f1_comoving test failed'

    all_prior = [[0.1, 0.5], [0, 3]] # Om, z
    npoints = 100
    seed = 2
    d = len(all_prior)

    lh_sampler = scipy.stats.qmc.LatinHypercube(d, seed=seed)
    samples = lh_sampler.random(npoints)
    for i in range(d):
        samples[:,i] = all_prior[i][0] + (all_prior[i][1] - all_prior[i][0]) * samples[:,i]

    Om = samples[:,0]
    z = samples[:,1]

    # Get truth
    a = 1/(1+z)
    x = (Om - 1) / Om * a ** 3
    lower = a**2 * np.sqrt(Om * a**(-3) + 1 - Om) * scipy.special.hyp2f1(2/3, 1, 7/6, x)
    x = (Om - 1) / Om
    upper = scipy.special.hyp2f1(2/3, 1, 7/6, x)
    rh = 2997.92458  # h^{-1} Mpc
    true = 2 * rh / Om * (upper - lower)

    pred = background.symbolic_radial_comoving(Om, z)

    # Test fractional error so take log
    true = np.log(true)
    pred = np.log(pred)
    assert np.allclose(pred, true, rtol=1e-4), 'symbolic_f test failed'

    return


def test_growth_hyper():
    
    all_prior = [[0.1, 0.5], [0, 3]] # Om, z
    npoints = 100
    seed = 2
    d = len(all_prior)

    lh_sampler = scipy.stats.qmc.LatinHypercube(d, seed=seed)
    samples = lh_sampler.random(npoints)
    for i in range(d):
        samples[:,i] = all_prior[i][0] + (all_prior[i][1] - all_prior[i][0]) * samples[:,i]

    Om = samples[:,0]
    z = samples[:,1]

    a = 1/(1+z)
    x = (Om - 1) / Om * a ** 3

    pred = linear.symbolic_D(Om, z)
    true = a * scipy.special.hyp2f1(1/3, 1, 11/6, x)

     # Test fractional error so take log
    true = np.log(true)
    pred = np.log(pred)
    assert np.allclose(pred, true, atol=5e-3), 'symbolic_D test failed'

    return


def test_growth_rate_hyper():

    all_prior = [[0.1, 0.5], [0, 3]] # Om, z
    npoints = 100
    seed = 2
    d = len(all_prior)

    lh_sampler = scipy.stats.qmc.LatinHypercube(d, seed=seed)
    samples = lh_sampler.random(npoints)
    for i in range(d):
        samples[:,i] = all_prior[i][0] + (all_prior[i][1] - all_prior[i][0]) * samples[:,i]

    Om = samples[:,0]
    z = samples[:,1]

    a = 1/(1+z)
    x = (Om - 1) / Om * a ** 3

    pred = linear.symbolic_f(Om, z)
    true = (1 + 6 * a ** 3 * (Om - 1) / (11 * Om) * 
        scipy.special.hyp2f1(4/3, 2, 17/6, x) /
        scipy.special.hyp2f1(1/3, 1, 11/6, x))

    # Test fractional error so take log
    true = np.log(true)
    pred = np.log(pred)
    assert np.allclose(pred, true, atol=2e-3), 'symbolic_f test failed'

    return


def get_cosmo():
    # Cosmological parameters
    As = 2.105 * 1e-9  # 10^9 A_s
    h = 0.6766
    Om = 0.3111
    Ob = 0.02242 / h ** 2
    ns = 0.9665
    tau = 0.0561
    return As, h, Om, Ob, ns, tau



def test_sigma8():

    As, h, Om, Ob, ns, tau = get_cosmo()
    sigma8 = linear.symbolic_sigma8(Om, Ob, h, ns, As)

    # Check inversion
    As_inverse = linear.symbolic_As(Om, Ob, h, ns, sigma8)
    assert np.isclose(As, As_inverse, rtol=1e-6), 'sigma8 <-> As inversion failed'
    
    # Get sigma8 with camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=0.0,
                       omk=0,
                       tau=tau,)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[0.], kmax=9)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb = results.get_sigma8()[0]
    assert np.isclose(sigma8, sigma8_camb, rtol=1e-3), 'symbolic_sigma8 does not match camb value'

    return


def test_pk_lin():

    As, h, Om, Ob, ns, tau = get_cosmo()
    sigma8 = linear.symbolic_sigma8(Om, Ob, h, ns, As)

    # Define k range
    kmin = 9e-3
    kmax = 9
    nk = 400
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # Get P(k) with camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=0.0,
                       omk=0,
                       tau=tau,)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[0.], kmax=9)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
        minkh=k.min(), maxkh=k.max(), npoints=len(k))
    pk_camb = pk_camb[0, :]

    # Get symbolic P(k)
    pk_sym = linear.symbolic_pklin(Om, Ob, h, ns, sigma8, 0.0, k)

    assert np.allclose(np.log(pk_sym), np.log(pk_camb), atol=1e-2), 'symbolic_pklin does not match camb P(k)'

    return



def test_pk_nonlin():

    As, h, Om, Ob, ns, tau = get_cosmo()
    sigma8 = linear.symbolic_sigma8(Om, Ob, h, ns, As)

    # Define k range
    kmin = 9e-3
    kmax = 9
    nk = 400
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    all_z = [0.0, 1.0, 2.0]

    # Get P(k) with camb
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100,
                       ombh2=Ob * h ** 2,
                       omch2=(Om - Ob) * h ** 2,
                       mnu=0.0,
                       omk=0,
                       tau=tau,)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=all_z, kmax=9)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version='takahashi')
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
        minkh=k.min(), maxkh=k.max(), npoints=len(k))

    # Get symbolic P(k)
    for i, z in enumerate(all_z):
        pk_sym = linear.symbolic_pklin(Om, Ob, h, ns, sigma8, z, k)
        ksigma = halofit.symbolic_ksigma(Om, Ob, h, ns, sigma8, z)
        neff = halofit.symbolic_neff(Om, Ob, h, ns, sigma8, z)
        C = halofit.symbolic_C(Om, Ob, h, ns, sigma8, z)
        pk_sym = halofit.apply_halofit(k, pk_sym, Om, Ob, h, ns, sigma8, z, ksigma, neff, C)   
        assert np.allclose(np.log(pk_sym), np.log(pk_camb[i]), atol=0.02), 'symbolic_pklin does not match camb P(k)'

    return
