#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_CHAR_PER_LINE 128
#define FLT_MAX 3.40282347e+38

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

float** file_read(char *filename, int  *num_objs, int  *num_coordinates)
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;

    FILE *infile;
    char *line, *ret;
    int   lineLen;

    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    lineLen = MAX_CHAR_PER_LINE;
    line = (char*) malloc(lineLen);

    (*num_objs) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        while (strlen(line) == lineLen-1) {
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);

            lineLen += MAX_CHAR_PER_LINE;
            line = (char*) realloc(line, lineLen);

            ret = fgets(line, lineLen, infile);
        }

        if (strtok(line, " \t\n") != 0)
            (*num_objs)++;
    }
    rewind(infile);

    (*num_coordinates) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            while (strtok(NULL, " ,\t\n") != NULL) (*num_coordinates)++;
            break;
        }
    }
    rewind(infile);
    len = (*num_objs) * (*num_coordinates);
    objects    = (float**)malloc((*num_objs) * sizeof(float*));
    objects[0] = (float*) malloc(len * sizeof(float));
    for (i=1; i<(*num_objs); i++)
        objects[i] = objects[i-1] + (*num_coordinates);

    i = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j=0; j<(*num_coordinates); j++)
            objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        i++;
    }

    fclose(infile);
    free(line);

    return objects;
}

static int cal_pow_2(int n) {
    int res = 0;
    while(n > 0){
        n >>= 1;
        res = (res<<1) | 1;
    }
    return (res+1);
}

__host__ __device__ static
float cal_dist(int num_coordinates, int num_objs, int num_clusters, float *objects, float *clusters, int objectId, int clusterId){
    float ans=0.0;

    for (int i = 0; i < num_coordinates; i++) {
        float temp = (objects[num_objs * i + objectId] - clusters[num_clusters * i + clusterId]);
        ans += pow(temp,2);
    }
    ans = sqrt(ans);
    return ans;
}

__global__ static
void find_nearest_cluster(int num_coordinates, int num_objs, int num_clusters, float *objects, float *dev_clusters, int *relationship,
                          int *curr_temporaries){
    extern __shared__ char sharedMemory[];

    unsigned char *relationshipChanged = (unsigned char *)sharedMemory;

    relationshipChanged[threadIdx.x] = 0;

    int objectId =  threadIdx.x + (blockDim.x * blockIdx.x);

    if (objectId < num_objs) {
        float min_dist;
        int index  = -1;
        min_dist = FLT_MAX;
        float *clusters = dev_clusters;
        for (int i=0; i<num_clusters; i++) {
            float dist = cal_dist(num_coordinates, num_objs, num_clusters,
                                 objects, clusters, objectId, i);
            index = (dist < min_dist ? (min_dist = dist, i): index);
        }

        if (relationship[objectId] != index) {
            relationship[objectId] = index;
            relationshipChanged[threadIdx.x] = 1;
        }

        __syncthreads();
        unsigned int s = blockDim.x / 2;
        while(s > 0) {
            relationshipChanged[threadIdx.x] += ((threadIdx.x < s) ? relationshipChanged[threadIdx.x + s] : 0);
            s >>= 1;
            __syncthreads();
        }
         
        if (!(threadIdx.x)) {
            curr_temporaries[blockIdx.x] = relationshipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *devicetemporaries, int numtemporaries, int numtemporaries2){
    
    numtemporaries2 >>= 1;
    extern __shared__ unsigned int curr_temporaries[];

    curr_temporaries[threadIdx.x] =
        ((threadIdx.x >= numtemporaries) ? 0 : devicetemporaries[threadIdx.x]);

    __syncthreads();
    
    unsigned int s =  numtemporaries2;
    while(s > 0) {
        curr_temporaries[threadIdx.x] += ((threadIdx.x < s) ? curr_temporaries[threadIdx.x + s] : 0);
        s >>= 1;
        __syncthreads();
    }

    if (!(threadIdx.x)) {
        devicetemporaries[0] = curr_temporaries[0];
    }
}

float** cuda_kmeans(float **objects, int num_coordinates, int num_objs, int num_clusters, int *relationship){

    float  **dimObjects;
    malloc2D(dimObjects, num_coordinates, num_objs, float);
    for (int i = 0; i < num_coordinates; i++) {
        for (int j = 0; j < num_objs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    float *dev_clusters;
    float  **dimClusters;
    malloc2D(dimClusters, num_coordinates, num_clusters, float);
    for (int i = 0; i < num_coordinates; i++) {
        for (int j = 0; j < num_clusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    memset(relationship, -1, num_objs*sizeof(int));

    int *newClusterSize; 
    newClusterSize = (int*) calloc(num_clusters, sizeof(int));

    float  **newClusters;
    malloc2D(newClusters, num_coordinates, num_clusters, float);
    memset(newClusters[0], 0, num_coordinates * num_clusters * sizeof(float));

    unsigned int numThreadsPerClusterBlock = 128;
    unsigned int numClusterBlocks =
        (num_objs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

    unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);

    unsigned int numReductionThreads =
        cal_pow_2(numClusterBlocks);
    unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);


    float *dev_objs;
    int *dev_relationship;
    int *devicetemporaries;

    cudaMalloc(&dev_objs, num_objs*num_coordinates*sizeof(float));
    cudaMalloc(&dev_clusters, num_clusters*num_coordinates*sizeof(float));
    cudaMalloc(&dev_relationship, num_objs*sizeof(int));
    cudaMalloc(&devicetemporaries, numReductionThreads*sizeof(unsigned int));

    cudaMemcpy(dev_objs, dimObjects[0], num_objs*num_coordinates*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_relationship, relationship, num_objs*sizeof(int), cudaMemcpyHostToDevice);

    for(int loop = 0; loop < 500; loop++){
        cudaMemcpy(dev_clusters, dimClusters[0], num_clusters*num_coordinates*sizeof(float), cudaMemcpyHostToDevice);

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (num_coordinates, num_objs, num_clusters,
             dev_objs, dev_clusters, dev_relationship, devicetemporaries);

        cudaDeviceSynchronize();

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (devicetemporaries, numClusterBlocks, numReductionThreads);

        cudaDeviceSynchronize();

        int d;
        cudaMemcpy(&d, devicetemporaries, sizeof(int), cudaMemcpyDeviceToHost);
        float delta = (float)d;

        cudaMemcpy(relationship, dev_relationship, num_objs*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i=0; i<num_objs; i++) {
            newClusterSize[relationship[i]] += 1;
            for (int j=0; j<num_coordinates; j++)
                newClusters[j][relationship[i]] += objects[i][j];
        }

        for (int i=0; i<num_clusters; i++) {
            for (int j=0; j<num_coordinates; j++) {
                if (newClusterSize[i] != 0)
                    dimClusters[j][i] = (newClusters[j][i] / (1.0*newClusterSize[i]));
                newClusters[j][i] = 0;
            }
            newClusterSize[i] = 0;
        }

        if(delta > 0.001){
            break;
        }
    }   

    float  **clusters;
    malloc2D(clusters, num_clusters, num_coordinates, float);
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < num_coordinates; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    return clusters;
}

int main(int argc, char **argv) {
    int     opt;

    int     num_clusters, num_coordinates, num_objs;
    int    *relationship;
    char   *filename;
    float **objects;
    float **clusters;
    num_clusters      = 0;
    filename         = NULL;

    while ( (opt=getopt(argc,argv,"i:n:"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'n': num_clusters = atoi(optarg);
                  break;
            case '?': 
                      break;
            default: 
                      break;
        }
    }
    struct timeval tvalBefore, tvalAfter;

    objects = file_read(filename, &num_objs, &num_coordinates);
    relationship = (int*) malloc(num_objs * sizeof(int));
    gettimeofday (&tvalBefore, NULL);

    clusters = cuda_kmeans(objects, num_coordinates, num_objs, num_clusters,
                          relationship);
    gettimeofday (&tvalAfter, NULL);

    printf("num_objs       = %d\n", num_objs);
    printf("num_coordinates     = %d\n", num_coordinates);
    printf("num_clusters   = %d\n", num_clusters);

    printf("Time: %ld microseconds\n",
        ((tvalAfter.tv_sec - tvalBefore.tv_sec)*1000000L
        +tvalAfter.tv_usec) - tvalBefore.tv_usec
        );

    return(0);
}