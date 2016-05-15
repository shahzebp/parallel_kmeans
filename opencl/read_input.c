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
int npoints = 0;
int nclusters = 5;
float threshold = 0.001;

float** kmeans_clustering(float **feature, int *membership)
{
    int i, j, n=0, loop=0, temp, *new_centers_len, *initial, initial_points;

	float delta, **clusters, **new_centers;

    if (nclusters > npoints)
        nclusters = npoints;

    clusters = (float**) malloc(nclusters * sizeof(float*));
    clusters[0] = (float*) malloc(nclusters * nfeatures * sizeof(float));
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

int setup(int argc, char **argv) {
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
