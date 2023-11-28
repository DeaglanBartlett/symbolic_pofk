# symbolic_pofk

A precise symbolic emulator of the linear matter power spectrum and for the conversion
$\sigma_8 \leftrightarrow A_{\rm s}$ as a function of cosmology.
Here we give the emulators as simple python functions, but these can be 
easily copied, pasted and modified to the language of your choice.
Please see [Bartlett et al. 2023](https://arxiv.org/abs/2311.15865) for further details.


## Installation

To install the emulators and the dependencies, run the following

```
git clone git@github.com:DeaglanBartlett/symbolic_pofk.git
pip install -e symbolic_pofk
```

## Examples

We give an example for how to use the emulators in `examples/examples.py`.

## Citation

If you use any of the emulators in this repository, please cite the following paper
```
@ARTICLE{symbolic_pofk,
     author = {{Bartlett}, D.~J. and {Kammerer}, L. and {Kronberger}, G. and {Desmond}, H.
               and {Ferreira}, P.~G. and {Wandelt} B.~D. and {Burlacu} B.
               and {Alonso}, D. and {Zennaro} M.},
      title = "{A precise symbolic emulator for the linear matter power spectrum}",
    journal = {arXiv e-prints},
   keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
       year = 2023,
      month = nov,
        eid = {arXiv:2311.15865},
      pages = {arXiv:2311.15865},
        doi = {10.48550/arXiv.2311.15865},
archivePrefix = {arXiv},
     eprint = {2311.15865},
primaryClass = {astro-ph.CO},
        url = {https://arxiv.org/abs/2311.15865},
}
```
