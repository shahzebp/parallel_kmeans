#include <stdio.h>
#include <time.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));
	if( argc < 3)
	{
		printf("Please supply 2 arguments\n");
		exit(0);
	}
	int vertices = atoi(argv[1]);
	int dimensions = atoi(argv[2]);
    float a = 10.0;
	for(int i=1;i<=vertices;i++)
	{
		printf("%d ", i);	
    	for (int j=0;j<dimensions;j++)
			printf("%f ", ((float)rand()/(float)(RAND_MAX)) * a + (-1* a/2));
		
		printf("\n");
	}
    return 0;
}
