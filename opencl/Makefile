# C compiler
CC = g++
CC_FLAGS = -g -O2
export OCL=$TACC_CUDA_DIR 
export LD_LIBRARY_PATH=/opt/apps/intel/opencl/lib64:$LD_LIBRARY_PATH

kmeans: read_input.o
	$(CC) $(CC_FLAGS) -lOpenCL kmeans.cpp read_input.o -o kmeans -I$OCL/include -L$LD_LIBRARY_PATH

kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) read_input.c -c

clean:
	rm -f *.o *~ kmeans *.linkinfo
