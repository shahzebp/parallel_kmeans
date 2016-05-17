#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include "kmeans.h"

int     numdims = 0;
float **objects;

static float cal_dist(float *coord1, float *coord2)
{
    int     i;
    float   ans = 0.0;

    for (i=0; i<numdims; i++) {

        float temp = (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
        ans += temp;
    }

    return(ans);
}

static int closest_cluster(int num_clusters, int num_dim, float  *object,
                         float **clusters)
{
    int   index = 0;
    float dist, min_dist;

    numdims  = num_dim;
    min_dist = cal_dist(object, clusters[0]);

    for (int i = 1; i < num_clusters; i++) {
        dist = cal_dist(object, clusters[i]);
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

float** seq_kmeans(int num_dim, int num_objs, int num_clusters, 
        int    *relationship)
{
    int     index, cnt = 0;
    int     *num_current_cluster;
    
    float threshold = 0.001;

    float    delta;
    float  **clusters;
    float  **curr_cluster;

    int i; 

    clusters    = (float**) malloc(num_clusters * sizeof(float*));
    clusters[0] = (float*)  malloc(num_clusters * num_dim * sizeof(float));
   
    for (int i = 1; i < num_clusters; i++)
        clusters[i] = clusters[i-1] + num_dim;

    for (int i = 0; i < num_clusters; i++)
        for (int j = 0; j < num_dim; j++)
            clusters[i][j] = objects[i][j];

    for (int i = 0; i<num_objs; i++)
        relationship[i] = -1;

    num_current_cluster = (int*) calloc(num_clusters, sizeof(int));

    curr_cluster    = (float**) malloc(num_clusters * sizeof(float*));
    
    curr_cluster[0] = (float*)  calloc(num_clusters * num_dim, sizeof(float));
    
    for (int i = 1; i<num_clusters; i++) {
        float * val = curr_cluster[i-1] + num_dim; 
        curr_cluster[i] = val;
    }

    do {
        delta = 0;
        for (int i = 0; i < num_objs; i++) {
            index = closest_cluster(num_clusters, num_dim, objects[i],
                                         clusters);
            if (relationship[i] != index) 
                delta++;

            relationship[i] = index;

            num_current_cluster[index]++;
            
            for (int j=0; j < num_dim; j++)
                curr_cluster[index][j] += objects[i][j];
        }

        delta = delta / num_objs;

        for (int i=0; i < num_clusters; i++) {
            for (int j = 0; j < num_dim; j++) {
                if (num_current_cluster[i] > 0) {
                    float val = curr_cluster[i][j] / num_current_cluster[i];
                    clusters[i][j] = val;
                }
                curr_cluster[i][j] = 0;
            }
            num_current_cluster[i] = 0;
        }
        
        if (delta < threshold)
            break;

    } while (cnt++ < 500);

    return clusters;
}

int main(int argc, char **argv) {
    
    float **clusters;
    int     num_clusters, num_dim, num_objs = 0;
    int    *relationship;
    char   *filename;
    int     opt;
    
    while ( (opt=getopt(argc,argv,"i:n:"))!= EOF) {
        switch (opt) {
            case 'i': filename = optarg;
                      break;
            case 'n': num_clusters = atoi(optarg);
                      break;
            default:
                      printf("Wrong option\n"); 
                      break;
        }
    }

    struct timeval tvalBefore, tvalAfter;

    objects = file_read(filename, &num_objs, &num_dim);
    
    gettimeofday (&tvalBefore, NULL);

    relationship = (int*) malloc(num_objs * sizeof(int));

    clusters = seq_kmeans(num_dim, num_objs, num_clusters,
                          relationship);

    gettimeofday (&tvalAfter, NULL);

    printf("num_objs       = %d\n", num_objs);
    printf("num_dim     = %d\n", num_dim);
    printf("num_clusters   = %d\n", num_clusters);

    printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );

   return(0);
}