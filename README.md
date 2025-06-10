# symbolic_pofk

[![arXiv](https://img.shields.io/badge/arXiv-2311.15865-b31b1b.svg)](https://arxiv.org/abs/2311.15865)
[![arXiv](https://img.shields.io/badge/arXiv-2402.17492-b31b1b.svg)](https://arxiv.org/abs/2402.17492)
[![arXiv](https://img.shields.io/badge/arXiv-2410.14623-b31b1b.svg)](https://arxiv.org/abs/2410.14623)
[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)

[![Coverage](https://img.shields.io/codecov/c/github/DeaglanBartlett/symbolic_pofk)](https://app.codecov.io/gh/DeaglanBartlett/symbolic_pofk)
[![Build Status](https://github.com/DeaglanBartlett/symbolic_pofk/actions/workflows/build-linux.yml/badge.svg)](https://github.com/DeaglanBartlett/symbolic_pofk/actions/workflows/build-linux.yml)
[![Build Status MacOS](https://github.com/DeaglanBartlett/symbolic_pofk/actions/workflows/build-mac.yml/badge.svg)](https://github.com/DeaglanBartlett/symbolic_pofk/actions/workflows/build-mac.yml)



Precise symbolic emulators of the linear and non-linear matter power spectra and for the conversion
$\sigma_8 \leftrightarrow A_{\rm s}$ as a function of cosmology.
Here we give the emulators as simple python functions and as a fortran90 routine, but these can be 
easily copied, pasted and modified to the language of your choice.
Please see
* [Bartlett et al. 2023](https://arxiv.org/abs/2311.15865) (Linear power spectrum) 
* [Bartlett et al. 2024](https://arxiv.org/abs/2402.17492) (Nonlinear power spectrum)
* [Sui et al. 2024](https://arxiv.org/abs/2410.14623) ($w_0-w_a$ and non-zero neutrino masses)
* [Kammerer et al. 2025](https://arxiv.org/abs/2410.14623) (Baryonic effects)
for further details.

For the linear and nonlinear LCDM emulators, by default, outside the k range tested in Bartlett et al. 2023, 
we use the Eisenstein & Hu fit which includes baryons. This can be switched off by setting
`extrapolate=False` in the functions `plin_emulated()`, `logF_max_precision()` and
`logF_fiducial()`. 


## Installation

To install the emulators and the dependencies, run the following

```
git clone git@github.com:DeaglanBartlett/symbolic_pofk.git
pip install symbolic_pofk
```

If you wish to use the pytorch code, the second line should be replaced with
```
pip install "symbolic_pofk[torch]"
```

If you wish to use the fortran version of the code, running the script
```
./setup_fortran.sh
```
will compile the fortran code and will produce a python wrapper for this.

## Examples

We give an example for how to use the ΛCDM
linear emulator in `examples/linear_example.py`.
and the nonlinear emulator in `examples/halofit_example.py`.
We show how to use the emulators for the extended cosmology
(including neutrino masses, w0 and wa) in `examples/syren_new_example.ipynb`.
And `examples/syren_baryon_example.ipynb` shows how to incorporate baryonic effects.

The example `examples/fortran_example.py` shows how to run the
fortran code with the python wrapper, and compares the difference
between the python and fortran implementations
(they are identical up to a fractional difference of
O(1e-6), which is much smaller than the error on the emulation).

## Citation

If you use any of the emulators in this repository, please cite the following paper
```
@ARTICLE{symbolic_pofk,
       author = {{Bartlett}, Deaglan J. and {Kammerer}, Lukas and {Kronberger}, Gabriel and {Desmond}, Harry and {Ferreira}, Pedro G. and {Wandelt}, Benjamin D. and {Burlacu}, Bogdan and {Alonso}, David and {Zennaro}, Matteo},
        title = "{A precise symbolic emulator of the linear matter power spectrum}",
      journal = {\aap},
     keywords = {methods: numerical, cosmological parameters, cosmology: theory, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing},
         year = 2024,
        month = jun,
       volume = {686},
          eid = {A209},
        pages = {A209},
          doi = {10.1051/0004-6361/202348811},
archivePrefix = {arXiv},
       eprint = {2311.15865},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...686A.209B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use the nonlinear emulator, please also cite the following paper
```
@ARTICLE{syren_halofit,
       author = {{Bartlett}, Deaglan J. and {Wandelt}, Benjamin D. and {Zennaro}, Matteo and {Ferreira}, Pedro G. and {Desmond}, Harry},
        title = "{SYREN-HALOFIT: A fast, interpretable, high-precision formula for the {\ensuremath{\Lambda}}CDM nonlinear matter power spectrum}",
      journal = {\aap},
     keywords = {methods: numerical, cosmological parameters, cosmology: theory, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing},
         year = 2024,
        month = jun,
       volume = {686},
          eid = {A150},
        pages = {A150},
          doi = {10.1051/0004-6361/202449854},
archivePrefix = {arXiv},
       eprint = {2402.17492},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...686A.150B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use the extended cosmologies, please cite
```
@ARTICLE{syren_new,
       author = {{Sui}, Ce and {Bartlett}, Deaglan J. and {Pandey}, Shivam and {Desmond}, Harry and {Ferreira}, Pedro G. and {Wandelt}, Benjamin D.},
        title = "{SYREN-NEW: Precise formulae for the linear and nonlinear matter power spectra with massive neutrinos and dynamical dark energy}",
      journal = {\aap},
     keywords = {methods: numerical, cosmological parameters, cosmology: theory, dark energy, large-scale structure of Universe, Cosmology and Nongalactic Astrophysics, Instrumentation and Methods for Astrophysics, Machine Learning, Neural and Evolutionary Computing},
         year = 2025,
        month = jun,
       volume = {698},
          eid = {A1},
        pages = {A1},
          doi = {10.1051/0004-6361/202452854},
archivePrefix = {arXiv},
       eprint = {2410.14623},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...698A...1S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use the baryonic emulators, please cite
```
@ARTICLE{syren_baryon,
     author = {{Kammerer}, L. and {Bartlett}, D.~J. and {Kronberger}, G. and {Desmond}, H.
               and {Ferreira}, P.~G. },  
      title = "{syren-new: Precise formulae for the linear and nonlinear matter power spectra 
      with massive neutrinos and dynamical dark energy}",
    journal = {arXiv e-prints},
   keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
       year = 2025,
      month = jun,
        eid = {arXiv:2410.14623},
      pages = {arXiv:2410.14623},
        doi = {10.48550/2410.14623},
archivePrefix = {arXiv},
     eprint = {2410.14623},
primaryClass = {astro-ph.CO},
        url = {https://arxiv.org/abs/2410.14623},
}
```

The software is available on the MIT licence:

Copyright 2024 Deaglan J. Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgements

DJB is supported by the Simons Collaboration on "Learning the Universe" and by 
Schmidt Sciences through The Eric and Wendy Schmidt AI in Science Fellowship.

