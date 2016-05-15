#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include "kmeans.h"

#define RANDOM_MAX 2147483647

float** kmeans_clustering(float **feature, int nfeatures, int npoints,
							int nclusters, float threshold, int *membership)
{    
    int      i, j, n = 0;
	int		 loop=0, temp;
    int     *new_centers_len;
    float    delta;
    float  **clusters;
    float  **new_centers;

	int     *initial;
	int      initial_points;
	int		 c = 0;

	if (nclusters > npoints)
		nclusters = npoints;

    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

	initial = (int *) malloc (npoints * sizeof(int));
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

    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

	do {
        delta = 0.0;
		delta = (float) kmeansOCL(feature, nfeatures, npoints, nclusters,
								membership, clusters, new_centers_len, new_centers);

		for (i=0; i<nclusters; i++) {
			for (j=0; j<nfeatures; j++) {
				if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;
			}
			new_centers_len[i] = 0;
		}	 
		c++;
    } while ((delta > threshold) && (loop++ < 500));
	printf("iterated %d times\n", c);
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}

