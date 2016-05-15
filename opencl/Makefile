CC = g++
CC_FLAGS = -g -O2
export OCL=$TACC_CUDA_DIR 
export LD_LIBRARY_PATH=/opt/apps/intel/opencl/lib64:$LD_LIBRARY_PATH

kmeans: kmeans.cpp
	$(CC) $(CC_FLAGS) -lOpenCL kmeans.cpp -o kmeans -I$OCL/include -L$LD_LIBRARY_PATH

clean:
	rm -f *.o *~ kmeans *.linkinfo
