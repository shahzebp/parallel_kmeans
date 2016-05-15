#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "kmeans.h"

int cluster(int npoints, int nfeatures, float **features, int min_nclusters, int max_nclusters,
					float threshold, int *best_nclusters, float ***cluster_centres, int nloops)
{    
	int		nclusters;						/* number of clusters k */	
	int		index =0;						/* number of iteration to reach the best RMSE */
	int		rmse;							/* RMSE for each clustering */
    int    *membership;						/* which cluster a data point belongs to */
    float **tmp_cluster_centres;			/* hold coordinates of cluster centers */
	int		i;

    membership = (int*) malloc(npoints * sizeof(int));

	for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
	{
		if (nclusters > npoints)
			break;

		allocate(npoints, nfeatures, nclusters, features);

		for(i = 0; i < nloops; i++)
		{
			tmp_cluster_centres = kmeans_clustering(features, nfeatures, npoints, nclusters, threshold, membership);
			if (*cluster_centres)
			{
				free((*cluster_centres)[0]);
				free(*cluster_centres);
			}
			*cluster_centres = tmp_cluster_centres;
			deallocateMemory();
		}
	}

    free(membership);

    return index;
}
