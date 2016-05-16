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

cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem cluster_dev;
cl_mem relationship_dev;

cl_kernel kernel_s, kernel2;

int   *membership_OCL, *membership_d;
float *feature_d, *clusters_d, *center_d;
int sourcesize = 1024*1024;

int nfeatures = 0;
int npoints = 0;
int nclusters = 5;
float threshold = 0.001;

float** kmeans_clustering(float **feature, int *membership)
{
    int i, j, n=0, loop=0, temp, *new_centers_len, *initial, initial_points;

	float delta, **clusters, **new_centers;

    if (nclusters > npoints)
        nclusters = npoints;

    size_t r1 = nclusters * sizeof(float*);
    clusters = (float**) malloc(r1);
    size_t r2 = nclusters * nfeatures * sizeof(float);
    clusters[0] = (float*) malloc(r2);

    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    initial = (int *)malloc(npoints * sizeof(int));
    
    for (i = 0; i < npoints; i++)
    {
        initial[i] = i;
    }

    initial_points = npoints;

    for (i=0; i<nclusters && initial_points >= 0; i++) {

        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[initial[n]][j];

        temp = initial[n];
        initial[n] = initial[initial_points-1];
        initial[initial_points-1] = temp;
        initial_points--;
        n++;
    }

    for (i=0; i < npoints; i++)
      membership[i] = -1;

  	size_t rawint = sizeof(int);

    new_centers_len = (int*) calloc(nclusters, );

    size_t nrawsize = nclusters *            sizeof(float*);
    new_centers    = (float**) malloc(nrawsize);

    size_t ncountrawsize = nclusters * nfeatures, sizeof(float);
    new_centers[0] = (float*)  calloc(ncountrawsize);

    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

    do {
        delta = 0.0;
        delta = (float) kmeansOCL(feature, nfeatures, npoints, nclusters,
                                membership, clusters, new_centers_len, new_centers);

        for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0) {
                	float temp = new_centers[i][j] / new_centers_len[i];
                    clusters[i][j] = temp;
                }
                new_centers[i][j] = 0;
            }
            new_centers_len[i] = 0;
        }
        
        if (delta < threshold)
            break;
    } while ((loop++ < 500));

    return clusters;
}

void cluster(float **features, float ***cluster_centres)
{
    int *membership;
    float **tmp_cluster_centres;

    membership = (int*) malloc(npoints * sizeof(int));

	if (nclusters > npoints)
		return;

	allocate(npoints, nfeatures, nclusters, features);

	tmp_cluster_centres = kmeans_clustering(features, membership);
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


int allocate(int n_points, int n_features, int n_clusters, float **feature)
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
	membership_OCL = (int*) malloc(n_points * sizeof(int));
	
    clReleaseProgram(prog);	
	
	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	
    d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	
    cluster_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err );
	
    relationship_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
		
	clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);
	
    clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, NULL, 0, 0, 0);	
}

int main( int argc, char** argv) 
{
	int opt, i, j;
		char   *filename = 0;
		char	line[1024];

		float *buf, **features, **cluster_centres=NULL;

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

    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        exit(1);
	}		
    while (fgets(line, 1024, infile) != NULL)
		if (strtok(line, " \t\n") != 0)
            npoints++;			
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first attribute): nfeatures = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
            break;
        }
    }        

    buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
    features    = (float**)malloc(npoints*          sizeof(float*));
    features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
    for (i=1; i<npoints; i++)
        features[i] = features[i-1] + nfeatures;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;            
        for (j=0; j<nfeatures; j++) {
            buf[i] = atof(strtok(NULL, " ,\t\n"));             
            i++;
        }            
    }
    fclose(infile);
	
	srand(7);
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float));

	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);

	cluster_centres = NULL;
    cluster(features, &cluster_centres);
    
    gettimeofday (&tvalAfter, NULL);

	printf("Coordinates of the Centroid are:\n");
	for(int l=0;l<nclusters;l++)
	{
		printf("Centroid Number %d: ", l);
		for(int m=0;m<nfeatures;m++)
			printf(" %0.4f", cluster_centres[l][m]);
		printf("\n");
	}

	printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );
    
    return(0);
}

int	kmeansOCL(float **feature, int n_features, int n_points, int n_clusters,
	int *membership, float **clusters, int *new_centers_len, float **new_centers)	
{
	size_t global_work[3] = { n_points, 1, 1 }; 

	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	
	clEnqueueWriteBuffer(cmd_queue, cluster_dev, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, 0);
					
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &cluster_dev);
	
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	int conv_point = 0;

	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);

	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &relationship_dev);
	int offset = 0;
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);

	int size = 0;
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, NULL, 0, 0, 0);
	clFinish(cmd_queue);
	clEnqueueReadBuffer(cmd_queue, relationship_dev, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
	
	
	for (int i = 0; i < n_points; i++)
	{
		new_centers_len[membership_OCL[i]]++;
		if (membership_OCL[i] != membership[i]) {
			membership[i] = membership_OCL[i];
			conv_point = conv_point + 1;
		}

		for (int j = 0; j < n_features; j++) {
			new_centers[membership_OCL[i]][j] += feature[i][j];
		}
	}

	return conv_point;
}