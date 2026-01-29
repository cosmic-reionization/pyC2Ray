FC=gfortran
F2PY=f2py
CXXFLAGS+=-DUSE_SUBBOX
F2PYFLAGS+=--f2cmap f2c.f2py_f2cmap $(CXXFLAGS)
FCFLAGS="-cpp"

all: libc2ray

libc2ray: photorates.f90 raytracing.f90 chemistry.f90
	$(F2PY) $(F2PYFLAGS) --f90flags=$(FCFLAGS) \
	--compiler=unix --fcompiler=gnu95 \
	-c photorates.f90 raytracing.f90 chemistry.f90 -m libc2ray

clean:
	rm -f *.so *.o *.mod