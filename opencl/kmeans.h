#ifndef HEADER_KMEANS
#define HEADER_KMEANS

int setup(int argc, char** argv);
int allocate(int npoints, int nfeatures, int nclusters, float **feature);
int	k_means_CL(float **feature, int nfeatures, int npoints, int nclusters, int *membership, float **clusters, int *new_centers_len, float  **new_centers);

#endif
