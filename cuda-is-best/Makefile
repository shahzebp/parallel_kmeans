all: cuda

DFLAGS      =
OPTFLAGS    = -O -NDEBUG
OPTFLAGS    = -g -pg
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v
LDFLAGS     = $(OPTFLAGS)
LIBS        =

CC          = gcc
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

CUDA_C_SRC = cuda_main.cu

CUDA_C_OBJ = $(CUDA_C_SRC:%.cu=%.o)

cuda: cuda_main
cuda_main: $(CUDA_C_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(CUDA_C_OBJ)

#---------------------------------------------------------------------
clean:
	rm -rf *.o cuda_main
