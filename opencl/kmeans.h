<<<<<<< HEAD
#define FLT_MAX 3.40282347e+38
=======
#ifndef _KMEANS
#define _KMEANS
>>>>>>> db787061f229d62667314980345e922711c1c665

int setup(int argc, char** argv);
int allocate(int npoints, int nfeatures, int nclusters, float **feature);
void deallocateMemory();
int	kmeansOCL(float **feature, int nfeatures, int npoints, int nclusters, int *membership, float **clusters, int *new_centers_len, float  **new_centers);


