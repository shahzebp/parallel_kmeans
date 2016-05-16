#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "kmeans.h"

static int nextPowerOfTwo(int n) {
    int res = 0;
    while(n > 0){
        n >>= 1;
        res = (res<<1) | 1;
    }
    return (res+1);
}

__host__ __device__ static
float euclid_dist_2(int numCoords, int numObjs, int numClusters, float *objects, float *clusters, int objectId, int clusterId){
    float ans=0.0;

    for (int i = 0; i < numCoords; i++) {
        float temp = (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
        ans += pow(temp,2);
    }
    ans = sqrt(ans);
    return ans;
}

__global__ static
void find_nearest_cluster(int numCoords, int numObjs, int numClusters, float *objects, float *deviceClusters, int *membership,
                          int *intermediates){
    extern __shared__ char sharedMemory[];

    unsigned char *membershipChanged = (unsigned char *)sharedMemory;

    membershipChanged[threadIdx.x] = 0;

    int objectId =  threadIdx.x + (blockDim.x * blockIdx.x);

    if (objectId < numObjs) {
        float min_dist;
        int index  = -1;
        min_dist = FLT_MAX;
        float *clusters = deviceClusters;
        for (int i=0; i<numClusters; i++) {
            float dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            index = (dist < min_dist ? (min_dist = dist, i): index);
        }

        if (membership[objectId] != index) {
            membership[objectId] = index;
            membershipChanged[threadIdx.x] = 1;
        }

        __syncthreads();
        unsigned int s = blockDim.x / 2;
        while(s > 0) {
            membershipChanged[threadIdx.x] += ((threadIdx.x < s) ? membershipChanged[threadIdx.x + s] : 0);
            s >>= 1;
            __syncthreads();
        }
         
        if (!(threadIdx.x)) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates, int numIntermediates, int numIntermediates2){
    
    numIntermediates2 >>= 1;
    extern __shared__ unsigned int intermediates[];

    intermediates[threadIdx.x] =
        ((threadIdx.x >= numIntermediates) ? 0 : deviceIntermediates[threadIdx.x]);

    __syncthreads();
    
    unsigned int s =  numIntermediates2;
    while(s > 0) {
        intermediates[threadIdx.x] += ((threadIdx.x < s) ? intermediates[threadIdx.x + s] : 0);
        s >>= 1;
        __syncthreads();
    }

    if (!(threadIdx.x)) {
        deviceIntermediates[0] = intermediates[0];
    }
}

float** cuda_kmeans(float **objects, int numCoords, int numObjs, int numClusters, int *membership){

    float  **dimObjects;
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    float *deviceClusters;
    float  **dimClusters;
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    memset(membership, -1, numObjs*sizeof(int));

    int *newClusterSize; 
    newClusterSize = (int*) calloc(numClusters, sizeof(int));

    float  **newClusters;
    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    unsigned int numThreadsPerClusterBlock = 128;
    unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;

    unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);

    unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);


    float *deviceObjects;
    int *deviceMembership;
    int *deviceIntermediates;

    cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float));
    cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float));
    cudaMalloc(&deviceMembership, numObjs*sizeof(int));
    cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int));

    cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice);

    for(int loop = 0; loop < 500; loop++){
        cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice);

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize();

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);

        cudaDeviceSynchronize();

        int d;
        cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost);
        float delta = (float)d;

        cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i=0; i<numObjs; i++) {
            newClusterSize[membership[i]] += 1;
            for (int j=0; j<numCoords; j++)
                newClusters[j][membership[i]] += objects[i][j];
        }

        for (int i=0; i<numClusters; i++) {
            for (int j=0; j<numCoords; j++) {
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
    malloc2D(clusters, numClusters, numCoords, float);
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    return clusters;
}

