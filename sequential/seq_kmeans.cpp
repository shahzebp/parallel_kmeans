#include <stdio.h>
#include <stdlib.h>

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
        float   threshold, int    *membership, int *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize;
                           
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

    for (i=0; i<numObjs; i++) membership[i] = -1;

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

    *loop_iterations = loop + 1;

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}
