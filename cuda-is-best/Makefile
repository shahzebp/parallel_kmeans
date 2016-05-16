.KEEP_STATE:

all: cuda 

DFLAGS      =
OPTFLAGS    = -O -NDEBUG
OPTFLAGS    = -g -pg
INCFLAGS    = -I.
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v
LDFLAGS     = $(OPTFLAGS)


CC          = gcc
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h

%.o : %.cu
	$(NVCC) -arch=compute_35 -code=sm_35 $@ -c $<

CUDA_C_SRC = cuda_main.cu cuda_io.cu cuda_wtime.cu
CUDA_CU_SRC = cuda_kmeans.cu

CUDA_C_OBJ = $(CUDA_C_SRC:%.cu=%.o)
CUDA_CU_OBJ = $(CUDA_CU_SRC:%.cu=%.o)

cuda: cuda_main
cuda_main: $(CUDA_C_OBJ) $(CUDA_CU_OBJ)
	$(NVCC) $(LDFLAGS) -o $@ $(CUDA_C_OBJ) $(CUDA_CU_OBJ)

#---------------------------------------------------------------------
clean:
	rm -rf *.o  cuda_main \
	       core* .make.state gmon.out     \
               *.cluster_centres *.membership