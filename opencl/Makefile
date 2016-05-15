# C compiler
CC = g++
CC_FLAGS = -g -O2
export OCL=$TACC_CUDA_DIR 
export LD_LIBRARY_PATH=/opt/apps/intel/opencl/lib64:$LD_LIBRARY_PATH

kmeans: cluster.o read_input.o kmeans_clustering.o
	$(CC) $(CC_FLAGS) -lOpenCL kmeans.cpp cluster.o read_input.o kmeans_clustering.o -o kmeans -I$OCL/include -L$LD_LIBRARY_PATH

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) read_input.c -c

clean:
	rm -f *.o *~ kmeans *.linkinfo
