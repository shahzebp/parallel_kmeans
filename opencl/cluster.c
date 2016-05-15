#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "kmeans.h"

float	min_rmse_ref = FLT_MAX;		
extern double wtime(void);
	/* reference min_rmse value */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,				/* number of data points */
            int      nfeatures,				/* number of attributes for each point */
            float  **features,			/* array: [npoints][nfeatures] */                  
            int      min_nclusters,			/* range of min to max number of clusters */
			int		 max_nclusters,
            float    threshold,				/* loop terminating factor */
            int     *best_nclusters,		/* out: number between min and max with lowest RMSE */
            float ***cluster_centres,		/* out: [best_nclusters][nfeatures] */
			float	*min_rmse,				/* out: minimum RMSE */
			int		 isRMSE,				/* calculate RMSE */
			int		 nloops					/* number of iteration for each number of clusters */
			)
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

