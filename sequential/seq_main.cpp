#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include "kmeans.h"

int main(int argc, char **argv) {
    int     opt;
    extern char   *optarg;
    extern int     optind;
    

    int     numClusters, numCoords, numObjs;
    int    *membership;
    char   *filename;
    float **objects;
    float **clusters;
    float   threshold;

    threshold        = 0.001;
    numClusters      = 0;
    filename         = NULL;

    while ( (opt=getopt(argc,argv,"i:n:"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            default:
                      printf("Wrong option\n"); 
                      break;
        }
    }

    struct timeval tvalBefore, tvalAfter;

    objects = file_read(0, filename, &numObjs, &numCoords);
    
    if (objects == NULL) exit(1);

    gettimeofday (&tvalBefore, NULL);

    membership = (int*) malloc(numObjs * sizeof(int));

    clusters = seq_kmeans(objects, numCoords, numObjs, numClusters, threshold,
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

