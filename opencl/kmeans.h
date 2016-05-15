#ifndef _KMEANS
#define _KMEANS

/* rmse.c */
int setup(int argc, char** argv);
int allocate(int npoints, int nfeatures, int nclusters, float **feature);
void deallocateMemory();
int	kmeansOCL(float **feature, int nfeatures, int npoints, int nclusters, int *membership, float **clusters, int *new_centers_len, float  **new_centers);

#endif


