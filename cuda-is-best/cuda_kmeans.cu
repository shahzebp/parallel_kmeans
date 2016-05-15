#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kmeans.h"

static int nextPowerOfTwo(int n) {
    int res = 0;
    while(n > 0){
        n >>= 1;
        res = (res<<1) | 1;
    }
    return res;
}

__host__ __device__ static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    float ans=0.0;

    for (int i = 0; i < numCoords; i++) {
        float temp = (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
        ans += temp;
    }
    ans = sqrt(ans);
    return ans;
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block. See numThreadsPerClusterBlock in cuda_kmeans().
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;

    float *clusters = deviceClusters;

    membershipChanged[threadIdx.x] = 0;

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            if (dist < min_dist) { 
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

void init(int numCoords, int numObjs, float **objects, int numClusters, float **dimObjects, float **dimClusters, int *membership, int *newClusterSize, float **newClusters){
    int i,j;
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));
}


float** cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   int    *membership   /* out: [numObjs] */
                   )
{
    int     *newClusterSize; 
    float    delta;
    float  **dimObjects;
    float  **clusters;
    float  **dimClusters;
    float  **newClusters;

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;


    //init(numCoords, numObjs, objects, numClusters, dimObjects, dimClusters, membership, newClusterSize, newClusters);

    malloc2D(dimObjects, numCoords, numObjs, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    malloc2D(dimClusters, numCoords, numClusters, float);
    for (int i = 0; i < numCoords; i++) {
        for (int j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    memset(membership, -1, numObjs*sizeof(int));

    newClusterSize = (int*) calloc(numClusters, sizeof(int));

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
        delta = (float)d;

        cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost);

        for (int i=0; i<numObjs; i++) {
            newClusterSize[membership[i]]++;
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

    malloc2D(clusters, numClusters, numCoords, float);
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    return clusters;
}


