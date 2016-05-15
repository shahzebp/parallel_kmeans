#define _CRT_SECURE_NO_DEPRECATE 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include "kmeans.h"
#include <unistd.h>
#include <sys/time.h>

float** kmeans_clustering(float **feature, int nfeatures, int npoints,
                            int nclusters, float threshold, int *membership)
{
    int      i, j, n = 0;
    int      loop=0, temp;
    int     *new_centers_len;
    float    delta;
    float  **clusters;
    float  **new_centers;

    int     *initial;
    int      initial_points;
    int      c = 0;

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

int cluster(int npoints, int nfeatures, float **features, int min_nclusters, int max_nclusters,
                    float threshold, int *best_nclusters, float ***cluster_centres, int nloops)
{
    int nclusters;
    int index =0;
    int *membership;
    float **tmp_cluster_centres;
    int i;

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

int setup(int argc, char **argv) {
		int		opt;
 extern char   *optarg;
		char   *filename = 0;
		float  *buf;
		char	line[1024];
		float	threshold = 0.001;		/* default value */
		int		max_nclusters=5;		/* default value */
		int		min_nclusters=5;		/* default value */
		int		best_nclusters = 0;
		int		nfeatures = 0;
		int		npoints = 0;
		float	len;
		         
		float **features;
		float **cluster_centres=NULL;
		int		i, j, index;
		int		nloops = 1;				/* default value */
				
		int		isOutput = 1;
		float	cluster_timing;

		while ( (opt=getopt(argc,argv,"i:t:m:n:l:o"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'm': max_nclusters = atoi(optarg);
                      break;
            case 'n': min_nclusters = atoi(optarg);
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

        /* allocate space for features[] and read attributes of all objects */
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
	
	printf("\nI/O completed\n");
	printf("\nNumber of objects: %d\n", npoints);
	printf("Number of features: %d\n", nfeatures);	
	
	if (npoints < min_nclusters)
	{
		printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
		exit(0);
	}

	srand(7);
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float));
	free(buf);

	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);

	cluster_centres = NULL;
    index = cluster(npoints, nfeatures, features, min_nclusters, max_nclusters,
					threshold, &best_nclusters, &cluster_centres, nloops);
    
    gettimeofday (&tvalAfter, NULL);


	if(min_nclusters == max_nclusters)
	{
		printf("\nCentroid Coordinates\n");
		for(i = 0; i < max_nclusters; i++)
		{
			printf("%d:", i);
			for(j = 0; j < nfeatures; j++)
				printf(" %.2f", cluster_centres[i][j]);
			printf("\n\n");
		}
	}

	len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

	printf("Number of Iteration: %d\n", nloops);

	printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

    printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );

	/* free up memory */
	free(features[0]);
	free(features);    
    return(0);
}
