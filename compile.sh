#!/bin/bash
cd external || exit
gfortran -c bvp_la-2.f -std=legacy
gfortran -c bvp_m-2.f90 -std=f95
cd ../src/superconductivity/multilayer || exit
python -m numpy.f2py -c -I../../../external/ ../../../external/bvp_la-2.o ../../../external/bvp_m-2.o -m pde pde.f90
