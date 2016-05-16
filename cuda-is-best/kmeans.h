#ifndef _H_KMEANS
#define _H_KMEANS

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

float** cuda_kmeans(float**, int, int, int, float, int*, int*);

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);

double  wtime(void);

extern int _debug;

#endif
