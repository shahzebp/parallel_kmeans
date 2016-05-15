
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#include <pthread.h>
#include <sys/time.h>
#include "kmeans.h"

double gettime() {
	struct timeval t;
	gettimeofday(&t,NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

#ifdef NV 
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#define FLT_MAX 3.40282347e+38

static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem d_cluster;
cl_mem d_membership;

cl_kernel kernel_s;
cl_kernel kernel2;

int   *membership_OCL;
int   *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

static int initialize(int use_gpu)
{
	cl_int result;
	size_t size;

	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { 
		return -1; 
	}
	
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};

	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	device_list = new cl_device_id[num_devices];

	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );

	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );

	return 0;
}


int allocate(int n_points, int n_features, int n_clusters, float **feature)
{

	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	char * tempchar = "./kmeans.cl";
	FILE * fp = fopen(tempchar, "rb"); 
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	
	cl_int err = 0;	
	int use_gpu = 1;

	if(initialize(use_gpu)) return -1;

	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

	char * kernel_kmeans_c  = "kmeans_kernel_c";
	char * kernel_swap  = "kmeans_swap";	
		
	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);  
	kernel2 = clCreateKernel(prog, kernel_swap, &err);  
		
	size_t global_work[3] = { n_points, 1, 1 };
	membership_OCL = (int*) malloc(n_points * sizeof(int));
	
    clReleaseProgram(prog);	
	
	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	
    d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	
    d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err );
	
    d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
		
	clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);
	
    clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, NULL, 0, 0, 0);
	
}

int main( int argc, char** argv) 
{
	setup(argc, argv);

}

int	kmeansOCL(float **feature, int n_features, int n_points, int n_clusters,
	int *membership, float **clusters, int *new_centers_len, float **new_centers)	
{
  
	int delta = 0;
	int i, j, k;
	cl_int err = 0;
	
	size_t global_work[3] = { n_points, 1, 1 }; 
	
	clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, 0);
	int size = 0; int offset = 0;
					
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);
	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);
	clFinish(cmd_queue);
	clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
	
	delta = 0;
	for (i = 0; i < n_points; i++)
	{
		int cluster_id = membership_OCL[i];
		new_centers_len[cluster_id]++;
		if (membership_OCL[i] != membership[i])
		{
			delta++;
			membership[i] = membership_OCL[i];
		}
		for (j = 0; j < n_features; j++)
		{
			new_centers[cluster_id][j] += feature[i][j];
		}
	}

	return delta;
}
