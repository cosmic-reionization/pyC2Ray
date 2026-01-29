CXX := g++
NVCC := nvcc

# Path to the GCC 13 libraries on COSMA
GCC_LIB := /cosma/local/gcc/13.1.0/lib64

# Include Paths
PYTHONINC := -I/cosma/local/Python/3.9.19/include/python3.9
# Path to NumPy include from your environment
NUMPYINC := -I/cosma8/data/dp004/dc-atta2/LW_in_pyc2ray/pyc2ray_implementation/pyc2ray-env/lib/python3.9/site-packages/numpy/_core/include

INC := $(PYTHONINC) $(NUMPYINC)

# Compiler Flags
# Using C++20 for std::source_location support
CUDAFLAGS := -std=c++20 -O2 -Xcompiler -fPIC -D PERIODIC -D LOCALRATES --gpu-architecture=sm_80

# Linker Flags
# -Xlinker passes the rpath to the host linker to avoid NVCC "unknown option" errors
# -lstdc++ explicitly links the modern C++ standard library
LDFLAGS := -L$(GCC_LIB) -Xlinker -rpath=$(GCC_LIB) -lstdc++

TARGET := libasora.so
# ADDED utils.o to the objects list
OBJS := memory.o rates.o raytracing.o python_module.o utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CUDAFLAGS) -shared -o $(TARGET) $(OBJS) $(LDFLAGS)

# Device code objects (require -dc for relocatable device code)
raytracing.o: raytracing.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc raytracing.cu -o raytracing.o

rates.o: rates.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc rates.cu -o rates.o

memory.o: memory.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc memory.cu -o memory.o

# ADDED rule for utils.o to define the safe_cuda symbol
utils.o: utils.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -dc utils.cu -o utils.o

# Host/Python wrapper (require -c)
python_module.o: python_module.cu
	$(NVCC) $(CUDAFLAGS) $(INC) -c python_module.cu -o python_module.o

clean:
	rm -f $(TARGET) *.o