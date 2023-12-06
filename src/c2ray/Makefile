FC=gfortran
F2PY=f2py
CXXFLAGS+=-DUSE_SUBBOX
F2PYFLAGS+=--f2cmap f2c.f2py_f2cmap $(CXXFLAGS)  #-DGREY_NOTABLES # Add -DGREY_NOTABLES to compile version with analytical formula for grey opacity rates
FCFLAGS="-cpp"

# Figure out the extension suffix for the platform (e.g. .cpython-310-x86_64-linux-gnu.so)
EXT_SUFFIX := $(shell python3-config --extension-suffix)

all: libc2ray

libc2ray: photorates.f90 raytracing.f90 chemistry.f90
	$(F2PY) $(F2PYFLAGS) --f90flags=$(FCFLAGS) -c photorates.f90 raytracing.f90 chemistry.f90 -m libc2ray
clean:
	rm *.so *.o *.mod
