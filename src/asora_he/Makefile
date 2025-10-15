HIPCC := hipcc

HIPCC_FLAGS := -O3 -fPIC -DPERIODIC -DLOCALRATES -fgpu-rdc -Wno-deprecated-register
HIPCC_LINK_FLAGS := -O3 -fgpu-rdc --hip-link

PY_INC := -I$(shell python -c "import sysconfig; print(sysconfig.get_path('include'))") \
          -I$(shell python -c "import numpy; print(numpy.get_include())")

all: libasora_He.so

libasora_He.so: memory_He.o rates.o raytracing_He.o python_module_He.o
	$(HIPCC) -shared $(HIPCC_LINK_FLAGS) -o $@ $+

raytracing_He.o: raytracing_He.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

rates.o: rates.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

memory_He.o: memory_He.cu
	$(HIPCC) $(HIPCC_FLAGS) -o $@ -c $<

python_module_He.o: python_module_He.cu
	$(HIPCC) $(HIPCC_FLAGS) $(PY_INC) -o $@ -c $<

clean:
	rm -f *.so *.o
