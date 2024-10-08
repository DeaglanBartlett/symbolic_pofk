# Copyright 2024 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
#  without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
import symbolic_pofk.linear as linear
import symbolic_pofk.syrenhalofit as syrenhalofit
import symbolic_pofk.f90_syrenhalofit as f90_syrenhalofit

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

# Redshift
z = 0.0
a = 1 / (1 + z)

# Compare sigma8 values
sigma8 = f90_syrenhalofit.emulators.as_to_sigma8(As, Om, Ob, h, ns)
print('\nf90 sigma8 = ', sigma8)
sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns)
print('py3 sigma8 = ', sigma8)

# Compare sigma8 values (old equation)
sigma8 = f90_syrenhalofit.emulators.old_as_to_sigma8(As, Om, Ob, h, ns)
print('\nf90 sigma8 (old eq) = ', sigma8)
sigma8 = linear.As_to_sigma8(As, Om, Ob, h, ns, old_equation=True)
print('py3 sigma8 (old eq) = ', sigma8)

# Compare As values
As_new = f90_syrenhalofit.emulators.sigma8_to_as(sigma8, Om, Ob, h, ns)
print('\nf90 As = ', As_new)
As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns)
print('py3 As = ', As_new)

# Compare As values (old equation)
As_new = f90_syrenhalofit.emulators.old_sigma8_to_as(sigma8, Om, Ob, h, ns)
print('\nf90 As (old eq) = ', As_new)
As_new = linear.sigma8_to_As(sigma8, Om, Ob, h, ns, old_equation=True)
print('py3 As (old eq) = ', As_new)

# Compare E&H formulae
EH_f90 = f90_syrenhalofit.emulators.eisenstein_hu_pk(k, sigma8, Om, Ob, h, ns)
EH_py3 = linear.pk_EisensteinHu_zb(k, sigma8, Om, Ob, h, ns)

# Compare logF corrections
F_f90 = np.exp(f90_syrenhalofit.emulators.logf_fiducial(
    k, sigma8, Om, Ob, h, ns))
F_py3 = np.exp(linear.logF_fiducial(
    k, sigma8, Om, Ob, h, ns, extrapolate=True))

#  Compare linear P(k)
pk_lin_f90 = f90_syrenhalofit.emulators.linear_pk_emulated(
    k, sigma8, Om, Ob, h, ns, a)
pk_lin_py3 = linear.plin_emulated(k, sigma8, Om, Ob, h, ns, a=a, emulator='fiducial',
                                  extrapolate=True)

# Compare 1+A correction
ksigma_f90 = f90_syrenhalofit.emulators.ksigma_emulated(
    sigma8, Om, Ob, h, ns, a)
neff_f90 = f90_syrenhalofit.emulators.neff_emulated(sigma8, Om, Ob, h, ns, a)
C_f90 = f90_syrenhalofit.emulators.c_emulated(sigma8, Om, Ob, h, ns, a)
A_f90 = f90_syrenhalofit.emulators.a_emulated(k, sigma8, Om, Ob, h, ns, a,
                                              ksigma_f90, neff_f90, C_f90)
ksigma_py3 = syrenhalofit.ksigma_emulated(sigma8, Om, Ob, h, ns, a)
neff_py3 = syrenhalofit.neff_emulated(sigma8, Om, Ob, h, ns, a)
C_py3 = syrenhalofit.C_emulated(sigma8, Om, Ob, h, ns, a)
A_py3 = syrenhalofit.A_emulated(k, sigma8, Om, Ob, h, ns, a,
                                ksigma=ksigma_py3, neff=neff_py3, C=C_py3)
print('\nf90 ksigma = ', ksigma_f90)
print('py3 ksigma = ', ksigma_py3)
print('\nf90 neff = ', neff_f90)
print('py3 neff = ', neff_py3)
print('\nf90 C = ', C_f90)
print('py3 C = ', C_py3)

# Compare non-linear P(k)
pk_f90 = f90_syrenhalofit.emulators.run_halofit(k, sigma8, Om, Ob, h, ns, a,
                                                'Bartlett', True)
pk_py3 = syrenhalofit.run_halofit(k, sigma8, Om, Ob, h, ns, a, emulator='fiducial',
                                  extrapolate=True, which_params='Bartlett', add_correction=True)

fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs = axs.flatten()
cmap = plt.get_cmap('Set1')
axs[0].semilogx(k, EH_f90 / EH_py3 - 1, color=cmap(0))
axs[0].set_title('Eisenstein & Hu')
axs[1].semilogx(k, F_f90 / F_py3 - 1, color=cmap(0))
axs[1].set_title('Bartlett et al. 2023 F correction')
axs[2].semilogx(k, pk_lin_f90 / pk_lin_py3 - 1, color=cmap(0))
axs[2].set_title('Linear Power Spectrum')
axs[3].semilogx(k, (1 + A_f90) / (1 + A_py3) - 1, color=cmap(0))
axs[3].set_title('Bartlett et al. 2024 1+A correction')
axs[4].semilogx(k, pk_f90 / pk_py3 - 1, color=cmap(0))
axs[4].set_title(
    'Nonlinear Power Spectrum\n(syren-halofit; Bartlett et al. 2024)')
for ax in axs:
    ax.set_ylabel('Fortran / Python - 1')
    ax.axhline(y=0, color='k')
    ax.set_xlabel(r'$k \ / \ h {\rm Mpc^{-1}}$')
axs[-1].remove()
fig.align_labels()
fig.tight_layout()
plt.show()
