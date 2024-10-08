#~/bin/bash

cd symbolic_pofk/fortran
gfortran -o syrenhalofit syrenhalofit.f90
f2py -c --fcompiler=gfortran -m f90_syrenhalofit syrenhalofit.f90
mv *.so ../
