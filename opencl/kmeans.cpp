#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include "kmeans.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>

#include <CL/cl.h>

#define FLT_MAX 3.40282347e+38

static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

cl_mem d_dimension;
cl_mem d_dimension_swap;
cl_mem cluster_dev;
cl_mem relationship_dev;

cl_kernel kernel_s, kernel2;

int   *relationship_OCL, *relationship_d;
float *dimension_d, *clusters_d, *center_d;
int sourcesize = 1024*1024;

int ndimensions = 0;
int npoints = 0;
int nclusters = 5;
float threshold = 0.001;

float** k_means_cluster_op(float **dimension, int *relationship)
{
    int i, j, n=0, loop=0, temp, *curr_dimensions_len, *initial, initial_points;

	float delta, **clusters, **curr_dimensions;

    if (nclusters > npoints)
        nclusters = npoints;

    size_t r1 = nclusters * sizeof(float*);
    clusters = (float**) malloc(r1);
    size_t r2 = nclusters * ndimensions * sizeof(float);
    clusters[0] = (float*) malloc(r2);

    for (i=1; i<nclusters; i++) {
    	float *temp = clusters[i-1] + ndimensions;
        clusters[i] = temp;
    }

    size_t rawnp = npoints * sizeof(int);
    initial = (int *)malloc(rawnp);

    initial_points = npoints;

    for (i = 0; i < npoints; i++)
        initial[i] = i;

    for (i=0; 0 <= initial_points && i<nclusters; i++) {
        for (j=0; j<ndimensions; j++) {
        	float temp = dimension[initial[n]][j];
            clusters[i][j] = temp;
        }

        int temp = initial[n];
        initial_points--;

        initial[n] = initial[initial_points];
        initial[initial_points] = temp;

        n = n + 1;
    }

  	memset(relationship, -1, npoints * sizeof(int));

  	size_t rawint = sizeof(int);

    curr_dimensions_len = (int*) calloc(nclusters, rawint);

    size_t nrawsize = nclusters *            sizeof(float*);
    curr_dimensions    = (float**) malloc(nrawsize);

    size_t ncountrawsize = sizeof(float);

    curr_dimensions[0] = (float*)  calloc(nclusters * ndimensions, ncountrawsize);

    for (i=1; i<nclusters; i++) {
    	float *temp = curr_dimensions[i-1] + ndimensions;
        curr_dimensions[i] = temp;
    }

    do {
        delta = 0;
        delta = (float) k_means_CL(dimension, ndimensions, npoints, nclusters,
                                relationship, clusters, curr_dimensions_len, curr_dimensions);

        for (int i=0; i<nclusters; i++) {
            for (int j=0; j<ndimensions; j++) {
                if (curr_dimensions_len[i] > 0) {
                	float temp = curr_dimensions[i][j] / curr_dimensions_len[i];
                    clusters[i][j] = temp;
                }
                curr_dimensions[i][j] = 0;
            }
            curr_dimensions_len[i] = 0;
        }

        if (delta < threshold)
            break;
    } while ((loop++ < 500));

    return clusters;
}

void cluster(float **dimensions, float ***cluster_centres)
{
    int *relationship;
    float **tmp_cluster_centres;

    relationship = (int*) malloc(npoints * sizeof(int));

	if (nclusters > npoints)
		return;

	allocate(npoints, ndimensions, nclusters, dimensions);

	tmp_cluster_centres = k_means_cluster_op(dimensions, relationship);
	*cluster_centres = tmp_cluster_centres;
}

static int initialize()
{
	cl_int result;
	size_t size;

	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { 
		return -1; 
	}
	
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};

	device_type = CL_DEVICE_TYPE_GPU;

	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	device_list = new cl_device_id[num_devices];

	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );

	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );

	return 0;
}


int allocate(int n_points, int n_dimensions, int n_clusters, float **dimension)
{

	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	char * tempchar = "./kmeans.cl";
	FILE * fp = fopen(tempchar, "rb"); 
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	
	cl_int err = 0;	
	if(initialize()) return -1;

	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

	char * kernel_kmeans_c  = "kmeans_kernel_c";
	char * kernel_swap  = "kmeans_swap";	
		
	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);  
	kernel2 = clCreateKernel(prog, kernel_swap, &err);  
		
	size_t global_work[3] = { n_points, 1, 1 };
	relationship_OCL = (int*) malloc(n_points * sizeof(int));
	
    clReleaseProgram(prog);	
	
	d_dimension = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_dimensions * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_dimension);
	
    d_dimension_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_dimensions * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_dimension_swap);
	
    cluster_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_dimensions  * sizeof(float), NULL, &err );
	
    relationship_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
		
	clEnqueueWriteBuffer(cmd_queue, d_dimension, 1, 0, n_points * n_dimensions * sizeof(float), dimension[0], 0, 0, 0);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_dimensions);
	
    clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, NULL, 0, 0, 0);	
}

int main( int argc, char** argv) 
{
	    FILE *infile;
	int opt, i, j;
		char   *filename = 0;
		char	line[1024];

		float *buf, **dimensions, **cluster_centres=NULL;

		while ( (opt=getopt(argc,argv,"i:m:"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'm': nclusters = atoi(optarg);
                      break;
            default: 
                      break;
        }
    }


    if (NULL == (infile = fopen(filename, "r"))) {
        return -1;
	}		
    while (NULL != fgets(line, 1024, infile))
		if (strtok(line, " \t\n") != 0)
            npoints++;			

    rewind(infile);
    
    while (NULL != fgets(line, 1024, infile)) {
        if (0 != strtok(line, " \t\n")) {
            while (strtok(NULL, " ,\t\n") != NULL) ndimensions++;
            break;
        }
    }        

    size_t rawf = npoints*ndimensions*sizeof(float);
    buf         = (float*) malloc(rawf);

    size_t rawfp = npoints*          sizeof(float*);
    dimensions    = (float**)malloc(rawfp);

    size_t rawnfp = npoints*ndimensions*sizeof(float);
    dimensions[0] = (float*) malloc(rawnfp);

    for (i=1; i<npoints; i++) {
    	float *temp = dimensions[i-1] + ndimensions;
        dimensions[i] = temp;
    }
    i = 0;

    rewind(infile);
    
    while (NULL != fgets(line, 1024, infile)) {
        if (NULL == strtok(line, " \t\n")) continue;            
        for (j=0; j<ndimensions; j++) {
            buf[i] = atof(strtok(NULL, " ,\t\n"));             
            i++;
        }            
    }

	memcpy(dimensions[0], buf, npoints*ndimensions*sizeof(float));

	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);

	cluster_centres = NULL;
    cluster(dimensions, &cluster_centres);
    
    gettimeofday (&tvalAfter, NULL);

	printf("Coordinates of the Centroid are:\n");
	for(int l=0;l<nclusters;l++)
	{
		printf("Centroid Number %d: ", l);
		for(int m=0;m<ndimensions;m++)
			printf(" %0.4f", cluster_centres[l][m]);
		printf("\n");
	}

	printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );
    fclose(infile);

    return(0);
}

int	k_means_CL(float **dimension, int n_dimensions, int n_points, int n_clusters,
	int *relationship, float **clusters, int *curr_dimensions_len, float **curr_dimensions)	
{
	size_t global_work[3] = { n_points, 1, 1 }; 

	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	
	clEnqueueWriteBuffer(cmd_queue, cluster_dev, 1, 0, n_clusters * n_dimensions * sizeof(float), clusters[0], 0, 0, 0);
					
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_dimension_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &cluster_dev);
	
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	int conv_point = 0;

	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_dimensions);

	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &relationship_dev);
	int offset = 0;
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);

	int size = 0;
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);
	clFinish(cmd_queue);
	clEnqueueReadBuffer(cmd_queue, relationship_dev, 1, 0, n_points * sizeof(int), relationship_OCL, 0, 0, 0);
	
	
	for (int i = 0; i < n_points; i++)
	{
		curr_dimensions_len[relationship_OCL[i]]++;
		if (relationship_OCL[i] != relationship[i]) {
			relationship[i] = relationship_OCL[i];
			conv_point = conv_point + 1;
		}

		for (int j = 0; j < n_dimensions; j++) {
			curr_dimensions[relationship_OCL[i]][j] += dimension[i][j];
		}
	}

	return conv_point;
}