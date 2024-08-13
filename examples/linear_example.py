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
import symbolic_pofk.linear as linear
import camb

# Define k range
kmin = 9e-3
kmin = 1e-4
kmax = 9
nk = 400
extrapolate_kmin = 2e-3
extrapolate_kmax = kmax
k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

# Cosmological parameters
As = 2.105  # 10^9 A_s
h = 0.6766
Om = 0.3111
Ob = 0.02242 / h ** 2
ns = 0.9665
tau = 0.0561

# Get sigma8 for this As
sigma8_old = linear.As_to_sigma8(As, Om, Ob, h, ns, old_equation=True)
print('sigma8 (old equation) = ', sigma8_old)
sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
print('sigma8 = ', sigma8)

# See what As you get in reverse
As_new = linear.sigma8_to_As(sigma8_old, Om, Ob, h, ns, old_equation=True)
print('As_new (old equation) = ', As_new)
As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns)
print('As_new = ', As_new)

# Emulate P(k)
pk_eh = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)
pk_eh_b = linear.pk_EisensteinHu_b(k, sigma8, Om, Ob, h, ns)
pk_fid = linear.plin_emulated(k, sigma8, Om, Ob, h, ns,
    emulator='fiducial', extrapolate=False, kmin=extrapolate_kmin, kmax=extrapolate_kmax)
pk_prec = linear.plin_emulated(k, sigma8, Om, Ob, h, ns,
    emulator='max_precision', extrapolate=False, kmin=extrapolate_kmin, kmax=extrapolate_kmax)
logF_eh_b = np.log(pk_eh_b / pk_eh)
logF_fid = linear.logF_fiducial(k, sigma8, Om, Ob, h, ns, extrapolate=False, kmin=extrapolate_kmin, kmax=extrapolate_kmax)
logF_prec = linear.logF_max_precision(k, sigma8, Om, Ob, h, ns, extrapolate=False, kmin=extrapolate_kmin, kmax=extrapolate_kmax)

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
logF_camb = np.log(pk_camb / pk_eh)

fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
cmap = plt.get_cmap('Set1')
axs[0].semilogx(k, pk_eh / pk_camb, label='Zero Baryon (Eisenstein & Hu 1998)', color=cmap(0))
axs[0].semilogx(k, pk_eh_b / pk_camb, label='Baryon (Eisenstein & Hu 1998)', color=cmap(1))
axs[0].semilogx(k, pk_fid / pk_camb, label='Fiducial (Bartlett et al. 2023)', color=cmap(2))
axs[0].semilogx(k, pk_prec / pk_camb, label='Max precision (Bartlett et al. 2023)', color=cmap(3))
axs[0].semilogx(k, pk_camb / pk_camb, label='camb', color=cmap(4), ls='--')
axs[1].semilogx(k, logF_eh_b, label='Baryon', color=cmap(1))
axs[1].semilogx(k, logF_fid, label='Fiducial', color=cmap(2))
axs[1].semilogx(k, logF_prec, label='Max precision', color=cmap(3))
axs[1].semilogx(k, logF_camb, label='camb', color=cmap(4), ls='--')
axs[0].legend()
axs[1].set_xlabel(r'$k \ / \ h {\rm \, Mpc}^{-1}$')
axs[0].set_ylabel(r'$P(k) / P_{\rm camb}(k)$')
axs[1].set_ylabel(r'$\log F$')
axs[1].axhline(0, color='k')
fig.align_labels()
fig.tight_layout()
fig.savefig('planck_2018_comparison.png', bbox_inches='tight')

fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
frac_error = np.abs((np.sqrt(pk_camb) - np.sqrt(pk_fid)) / np.sqrt(pk_camb)) * 100
ax2.semilogx(k, frac_error, label='Bartlett et al. 2023', color=cmap(2))
ax2.legend()
for y in [0.5, 1.0, 1.5]:
    ax2.axhline(y, color='k', ls='--')
ax2.set_xlabel(r'$k \ / \ h {\rm \, Mpc}^{-1}$')
ax2.set_ylabel(r'$\left| \frac{T_{\rm camb} - T_{\rm fit}}{T_{\rm camb}} \right| \times 100$', fontsize=14)
ax2.set_title('Planck 2018 Best-Fit Cosmology')
ax2.set_ylim(0, None)
ax2.axvline(extrapolate_kmin, color='k', ls='--')
ax2.axvline(extrapolate_kmax, color='k', ls='--')
fig2.tight_layout()
fig2.savefig('planck_2018_errors.png', bbox_inches='tight')

plt.show()
