CXX=g++
NVCC=nvcc
PYTHONINC+=-I/opt/cray/pe/python/3.11.5/include/python3.11
NUMPYINC+=-I/users/mibianco/myvenv/pyc2ray-env/lib/python3.11/site-packages/numpy/core/include

# Include Paths for both local and piz daint
INC+=$(PYTHONINC) $(NUMPYINC)
CXXFLAGS+=-std=c++14 -O2 -pedantic -fPIC -D PERIODIC -D RATES #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
CUDAFLAGS+=-std=c++14 -O2  -Xcompiler -fPIC -D PERIODIC -D LOCALRATES --gpu-architecture=sm_80 #-Wall -Wextra -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused

all: libasora.so

libasora.so: memory.o rates.o raytracing.o python_module.o
	$(NVCC) $(CUDAFLAGS) $(INC) -shared -o libasora.so memory.o rates.o raytracing.o python_module.o

raytracing.o: raytracing.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing.cu

rates.o: rates.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc rates.cu

memory.o: memory.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc memory.cu

python_module.o: python_module.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -c python_module.cu

clean:
	rm *.so *.o
