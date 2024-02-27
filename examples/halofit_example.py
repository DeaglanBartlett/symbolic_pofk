# Copyright 2024 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
import camb
import symbolic_pofk.syrenhalofit as syrenhalofit
import symbolic_pofk.linear as linear
from matplotlib import rcParams
rcParams["text.usetex"] = True

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
sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
print('sigma8 = ', sigma8)

# Emulate P(k)
pk_halofit = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a, emulator='fiducial',
    extrapolate=True, which_params='Bartlett', add_correction=False)
pk_syrenhalofit = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a,
    emulator='fiducial', extrapolate=True, which_params='Bartlett', add_correction=True)
pk_lin = linear.plin_emulated(k, sigma8, Om, Ob, h, ns, a=a, emulator='fiducial',
    extrapolate=True)
    
# Run CAMB versions
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
kh, z, pk_camb_halofit = results.get_matter_power_spectrum(minkh=k[0], maxkh=k[-1], npoints=nk)
pk_camb_halofit = pk_camb_halofit[0]

pars.NonLinear = camb.model.NonLinear_both
pars.NonLinearModel.set_params(halofit_version='mead2020')
results = camb.get_results(pars)
nk = len(k)
kh, z, pk_camb_hmcode = results.get_matter_power_spectrum(minkh=k[0], maxkh=k[-1], npoints=nk)
pk_camb_hmcode = pk_camb_hmcode[0]

# Euclid emulator
try:
    import euclidemu2
    ee2 = euclidemu2.PyEuclidEmulator()
    euclid_cosmo_par = {
        'As':As,
        'Omm':Om,
        'Omb':Ob,
        'h':h,
        'ns':ns,
        'mnu':mnu,
        'w0':-1.,
        'wa':0.0
    }
    redshift = 1 / a - 1
    kh, pk_euclid, _, _ = ee2.get_pnonlin(euclid_cosmo_par, [redshift], k)
    pk_euclid = pk_euclid[0]
except ImportError:
    print('Euclid Emulator not available')
    pk_euclid = None
    
# BACCO emulator
try:
    import sys
    import os
    import contextmanager
    
    @contextmanager
    def suppress_stdout_stderr():
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = stdout
            sys.stderr = stderr

    with suppress_stdout_stderr():
        import baccoemu
        bacco_emulator = baccoemu.Matter_powerspectrum()
        
    params = {
        'omega_cold'    :  Om,
        'sigma8_cold'   :  sigma8, # if A_s is not specified
        'omega_baryon'  :  Ob,
        'ns'            :  ns,
        'hubble'        :  h,
        'neutrino_mass' :  mnu,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  a
    }
    
    _, pk_bacco = bacco_emulator.get_nonlinear_pk(k=k, cold=True, **params)
    
except ImportError:
    print('BACCO Emulator not available')
    pk_bacco = None
    
    
# Plot all the emulators
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
lw = 2
cmap = plt.get_cmap('Set1')

ax.loglog(k, pk_lin, color='k', ls='-.', label='Linear')
ax.loglog(k, pk_camb_halofit, color=cmap(1), ls=':',
    label=r'\textsc{halofit} (Takahashi+ 2012; \textsc{camb})', lw=lw)
ax.loglog(k, pk_camb_hmcode, color=cmap(4), ls=':',
    label=r'\textsc{hmcode} (Mead+ 2021; \textsc{camb})', lw=lw)
ax.loglog(k, pk_halofit, color=cmap(2), ls='--',
    label=r'\textsc{halofit+} (Bartlett et al. 2024)', lw=lw)
ax.loglog(k, pk_syrenhalofit, color=cmap(3), ls='--',
    label=r'\textsc{syren-halofit} (Bartlett et al. 2024)', lw=lw)
if pk_euclid is not None:
    ax.loglog(k, pk_euclid, color=cmap(0), ls ='-',
        label=r'\textsc{euclidemulator2}', lw=lw)
if pk_bacco is not None:
    ax.loglog(k, pk_bacco, color=cmap(6), ls ='-',
        label=r'\textsc{bacco} Emulator', lw=lw)
ax.set_ylabel(r'$P(k) \ / \ (h^{-1} {\rm \, Mpc})^3 $')
ax.set_xlabel(r'$k \ / \ h {\rm \, Mpc^{-1}}$')
ax.legend(framealpha=0.0, fontsize=12)
ax.set_title(r'$z = ' + str(int(redshift)) + '$')
fig.tight_layout()
plt.show()
