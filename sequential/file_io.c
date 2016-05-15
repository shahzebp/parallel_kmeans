#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "kmeans.h"

#define MAX_CHAR_PER_LINE 128

float** file_read(int   isBinaryFile, char *filename, int  *numObjs, int  *numCoords)
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;

    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        numBytesRead = read(infile, numObjs,    sizeof(int));
        numBytesRead = read(infile, numCoords, sizeof(int));

        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        objects[0] = (float*) malloc(len * sizeof(float));
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        numBytesRead = read(infile, objects[0], len*sizeof(float));

        close(infile);
    }
    else {
        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }

        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            while (strlen(line) == lineLen-1) {
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);

                ret = fgets(line, lineLen, infile);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);

        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break;
            }
        }
        rewind(infile);
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        objects[0] = (float*) malloc(len * sizeof(float));
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
            i++;
        }

        fclose(infile);
        free(line);
    }

    return objects;
}
