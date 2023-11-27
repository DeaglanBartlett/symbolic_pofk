import numpy as np
import matplotlib.pyplot as plt
import symbolic_pofk.emulators as emulators

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

# Get sigma8 for this As
sigma8 = emulators.As_to_sigma8(As, Om, Ob, h, ns)
print('sigma8 = ', sigma8)

# See what As you get in reverse
As_new = emulators.sigma8_to_As(sigma8, Om, Ob, h, ns)
print('As_new = ', As_new)

# Emulate P(k)
pk_eh = emulators.pk_EisensteinHu(k, sigma8, Om, Ob, h, ns)
pk_fid = emulators.plin_emulated(k, sigma8, Om, Ob, h, ns, emulator='fiducial')
pk_prec = emulators.plin_emulated(k, sigma8, Om, Ob, h, ns, emulator='max_precision')
logF_fid = emulators.logF_fiducial(k, sigma8, Om, Ob, h, ns)
logF_prec = emulators.logF_max_precision(k, sigma8, Om, Ob, h, ns)

fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
cmap = plt.get_cmap('Set1')
axs[0].loglog(k, pk_eh, label='Zero Baryon (Eisenstein & Hu 1998)', color=cmap(0))
axs[0].loglog(k, pk_fid, label='Fiducial (Bartlett et al. 2023)', color=cmap(1))
axs[0].loglog(k, pk_prec, label='Max precision (Bartlett et al. 2023)', color=cmap(2))
axs[1].semilogx(k, logF_fid, label='Fiducial', color=cmap(1))
axs[1].semilogx(k, logF_prec, label='Max precision', color=cmap(2))
axs[0].legend()
axs[1].set_xlabel(r'$k \ / \ h {\rm \, Mpc}^{-1}$')
axs[0].set_ylabel(r'$P(k) \ / \ ({\rm \, Mpc} / h)^3$')
axs[1].set_ylabel(r'$\log F$')
axs[1].axhline(0, color='k')
fig.align_labels()
fig.tight_layout()
plt.show()
