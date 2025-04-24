# Compiler flags for ALPHA
#F90FLAGS = -assume big_endian -C 
#F77FLAGS = -assume big_endian -extend_source -C 

# Compiler flags for SUN 
#F77FLAGS = -e -O5 -native  -fnonstd -xld -C -fast
#F90FLAGS = -e -O5 -native  -fnonstd -C -fast

# Compiler flags for DECSTATION
#F77 = -extend_source -assume big_endian  -O3

# Compiler flags for Silicon Graphics Power Challenge
# R4400
#F77 = -extend_source -O  -C
# R10000
#F77 = -extend_source  -64 -r10000 -O3 #-C -g -d_lines
#-static -OPT:roundoff=3:IEEE_arith=3 -lfastm

#Compiler options suitable fo cdml
#FC = pgf90
#F90FLAGS = -C -Mextend -byteswapio -fast -Mbounds -Ktrap=fp
#F77FLAGS = -C -Mextend -byteswapio -fast -Mbounds -Ktrap=fp

#for JUMP
#FC = xlf90_r
#F90FLAGS = -bmaxdata:0x80000000 -qsmp=omp #-convert big_endian
#F77FLAGS = -bmaxdata:0x80000000 -qsmp=omp #-convert big_endian

#for erebos
#FC = ifort
#F90FLAGS = -convert big_endian -warn all -fast -check -mcmodel medium -shared-intel
##F90FLAGS = -convert big_endian -warn all -debug -g -check -mcmodel medium -shared-intel
#F77FLAGS = -fast

#for erebos
FC = ifort
F90FLAGS = -check bounds -warn unused -parallel -openmp -convert little_endian
##F90FLAGS = -check bounds -warn unused -mcmodel medium -shared-intel -parallel -fpp -openmp
F77FLAGS = 

.SUFFIXES:
.SUFFIXES:  .exe .o .for .f .f90

.for.o:
	$(FC) -c $< -o $*.o $(F77FLAGS)  

.f.o:
	$(FC) -c $< -o $*.o $(F77FLAGS)  

.o.exe:
	$(FC) $< -o $*.exe $(F90FLAGS)   

.f90.o:
	$(FC) -c $< -o $*.o $(F90FLAGS)
clean:
	rm -rf *.o *.mod *exe

cic.exe: globalvars.o get_gadget_file.o cic.o cic_dens.o write_out_cic.o
	$(FC) globalvars.o get_gadget_file.o cic.o cic_dens.o write_out_cic.o -o cic.exe $(F90FLAGS)