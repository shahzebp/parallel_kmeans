#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#define MAX_CHAR_PER_LINE 128
#include <assert.h>
#define FLT_MAX 3.40282347e+38

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

float** cuda_kmeans(float**, int, int, int, int*);

double  wtime(void);

float** file_read(int   isBinaryFile,  /* flag: 0 or 1 */
                  char *filename,      /* input file name */
                  int  *numObjs,       /* no. data objects (local) */
                  int  *numCoords)     /* no. coordinates */
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

    /* first find the number of objects */
    lineLen = MAX_CHAR_PER_LINE;
    line = (char*) malloc(lineLen);
    assert(line != NULL);

    (*numObjs) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen-1) {
            /* this line read is not complete */
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);

            /* increase lineLen */
            lineLen += MAX_CHAR_PER_LINE;
            line = (char*) realloc(line, lineLen);
            assert(line != NULL);

            ret = fgets(line, lineLen, infile);
            assert(ret != NULL);
        }

        if (strtok(line, " \t\n") != 0)
            (*numObjs)++;
    }
    rewind(infile);

    /* find the no. objects of each object */
    (*numCoords) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first coordiinate): numCoords = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
            break; /* this makes read from 1st object */
        }
    }
    rewind(infile);
    /* allocate space for objects[][] and read all objects */
    len = (*numObjs) * (*numCoords);
    objects    = (float**)malloc((*numObjs) * sizeof(float*));
    assert(objects != NULL);
    objects[0] = (float*) malloc(len * sizeof(float));
    assert(objects[0] != NULL);
    for (i=1; i<(*numObjs); i++)
        objects[i] = objects[i-1] + (*numCoords);

    i = 0;
    /* read all objects */
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j=0; j<(*numCoords); j++)
            objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        i++;
    }

    fclose(infile);
    free(line);

    return objects;
}

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


double wtime(void) 
{
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}

#ifdef _TESTING_
int main(int argc, char **argv) {
    double time;

    time = wtime();
    printf("time of day = %10.4f\n", time);

    return 0;
}
#endif

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

int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     isBinaryFile, is_output_timing;

           int     numClusters, numCoords, numObjs;
           int    *membership;
           char   *filename;
           float **objects;
           float **clusters;
           float   threshold;
           double  timing, io_timing, clustering_timing;
           int     loop_iterations;

    threshold        = 0.001;
    numClusters      = 0;
    filename         = NULL;

    while ( (opt=getopt(argc,argv,"p:i:n:t:abdo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'n': numClusters = atoi(optarg);
                  break;
            case '?': 
                      break;
            default: 
                      break;
        }
    }

    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }

    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    clusters = cuda_kmeans(objects, numCoords, numObjs, numClusters,
                          membership);

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

    printf("numObjs       = %d\n", numObjs);
    printf("numCoords     = %d\n", numCoords);
    printf("numClusters   = %d\n", numClusters);

    printf("Computation timing = %10.4f sec\n", clustering_timing);

    return(0);
}

