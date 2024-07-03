HIPCC := hipcc

HIPCC_FLAGS := -O3 -fPIC -DPERIODIC -DLOCALRATES -fgpu-rdc -Wno-deprecated-register
HIPCC_LINK_FLAGS := -O3 -fgpu-rdc --hip-link

PY_INC := -I$(shell python -c "import sysconfig; print(sysconfig.get_path('include'))") \
          -I$(shell python -c "import numpy; print(numpy.get_include())")

all: libasora.so

libasora.so: memory.o rates.o raytracing.o python_module.o
	$(HIPCC) -shared $(HIPCC_LINK_FLAGS) -o $@ $+

raytracing.o: raytracing.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

rates.o: rates.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

memory.o: memory.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

python_module.o: python_module.cu
	$(HIPCC) $(HIPCC_FLAGS) $(PY_INC) -o $@ -c $<

clean:
	rm -f *.so *.o
