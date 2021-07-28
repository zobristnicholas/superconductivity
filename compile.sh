#!/bin/bash
cd external || exit
rm ./*.o
rm ./*.mod
gfortran -c bvp_la-2.f -std=legacy
gfortran -c bvp_m-2.f90 -std=f95
gfortran -c pchip.f90 -std=legacy
cd ../src/superconductivity/multilayer || exit
rm ./*.so
python -m numpy.f2py -lgomp --f90flags=-fopenmp -c \
  -I../../../external/ \
  ../../../external/pchip.o \
  ../../../external/bvp_la-2.o \
  ../../../external/bvp_m-2.o \
  -m usadel usadel.f90
