#~/bin/bash

cd fortran
gfortran syrenhalofit.f90
f2py -c --fcompiler=gfortran -m f90_syrenhalofit syrenhalofit.f90
mv *.so ../symbolic_pofk
