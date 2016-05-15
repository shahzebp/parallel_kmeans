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

int nfeatures = 0;

float** kmeans_clustering(float **feature, int npoints,
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
        if (delta < threshold)
            break;
    } while ((loop++ < 500));

    return clusters;
}

int cluster(int npoints, float **features, int nclusters,
									float threshold, float ***cluster_centres)
{
    int index =0;
    int *membership;
    float **tmp_cluster_centres;
    int i;

    membership = (int*) malloc(npoints * sizeof(int));

	if (nclusters > npoints)
		return 0;

	allocate(npoints, nfeatures, nclusters, features);

	tmp_cluster_centres = kmeans_clustering(features, npoints,
							nclusters, threshold, membership);
	*cluster_centres = tmp_cluster_centres;

    return index;
}

int setup(int argc, char **argv) {
		int		opt;
		char   *filename = 0;
		float  *buf;
		char	line[1024];
		float	threshold = 0.001;
		int		nclusters=5;
		int		npoints = 0;

		float **features;
		float **cluster_centres=NULL;
		int		i, j, index;

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
	
	printf("\nNumber of objects: %d\n", npoints);
	printf("Number of coordinates: %d\n", nfeatures);	
	
	srand(7);
	memcpy(features[0], buf, npoints*nfeatures*sizeof(float));
	free(buf);

	struct timeval tvalBefore, tvalAfter;
    gettimeofday (&tvalBefore, NULL);

	cluster_centres = NULL;
    index = cluster(npoints, features, nclusters,
					threshold, &cluster_centres);
    
    gettimeofday (&tvalAfter, NULL);


	printf("\nCentroid Coordinates\n");
	for(i = 0; i < nclusters; i++)
	{
		printf("%d:", i);
		for(j = 0; j < nfeatures; j++)
			printf(" %.2f", cluster_centres[i][j]);
		printf("\n\n");
	}

    printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );

	free(features[0]);
	free(features);    
    return(0);
}
