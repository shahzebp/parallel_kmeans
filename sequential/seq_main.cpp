#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include "kmeans.h"

int numdims = 0;

static float euclid_dist_2(float *coord1, float *coord2)
{
    int     i;
    float   ans = 0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

static int find_nearest_cluster(int numClusters, int numCoords, float  *object,
                         float **clusters)
{
    int   index = 0;
    float dist, min_dist;

    numdims  = numCoords;
    min_dist = euclid_dist_2(object, clusters[0]);

    for (int i = 1; i < numClusters; i++) {
        dist = euclid_dist_2(object, clusters[i]);
        if (dist >= min_dist) {
            continue;
        }
        else {
            index    = i;
            min_dist = dist;
        }
    }

    return (index);
}

float** seq_kmeans(float **objects, int numCoords, int numObjs, int numClusters, 
        int    *membership)
{
    int      i, j, index, loop=0;
    int     *newClusterSize;
    
    float threshold = 0.001;

    float    delta;
    float  **clusters;
    float  **newClusters;

    clusters    = (float**) malloc(numClusters * sizeof(float*));
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
   
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];

    for (i=0; i<numObjs; i++)
        membership[i] = -1;

    newClusterSize = (int*) calloc(numClusters, sizeof(int));

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    
    for (i = 1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    do {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);

            if (membership[i] != index) delta += 1.0;

            membership[i] = index;

            newClusterSize[index]++;
            
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }

        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;
            }
            newClusterSize[i] = 0;
        }
            
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

        return clusters;
}

int main(int argc, char **argv) {
    extern char   *optarg;
    extern int     optind;
    
    int     numClusters, numCoords, numObjs = 0;
    int    *membership;
    char   *filename;
    float **objects;
    float **clusters;

    int     opt;
    while ( (opt=getopt(argc,argv,"i:n:"))!= EOF) {
        switch (opt) {
            case 'i': filename = optarg;
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            default:
                      printf("Wrong option\n"); 
                      break;
        }
    }

    struct timeval tvalBefore, tvalAfter;

    objects = file_read(filename, &numObjs, &numCoords);
    
    gettimeofday (&tvalBefore, NULL);

    membership = (int*) malloc(numObjs * sizeof(int));

    clusters = seq_kmeans(objects, numCoords, numObjs, numClusters,
                          membership);

    gettimeofday (&tvalAfter, NULL);

    printf("numObjs       = %d\n", numObjs);
    printf("numCoords     = %d\n", numCoords);
    printf("numClusters   = %d\n", numClusters);

    printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );

   return(0);
}
