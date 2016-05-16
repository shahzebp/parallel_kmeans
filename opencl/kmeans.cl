#define FLT_MAX 3.40282347e+38

__kernel void
kmeans_kernel_c(__global float  *feature, __global float *clusters, __global int *membership, int npoints,
				int nclusters, int nfeatures, int offset, int size) 
{
	int point_id = get_global_id(0);
    
	float min_dist = FLT_MAX;

	if (point_id >=  npoints)
		return;

	int index = 0;

	for (int i = 0; i < nclusters; i++)
	{
		float ans  = 0;
		for (int l = 0; l<nfeatures; l++)
			ans += pow(feature[l*npoints+point_id]-clusters[i*nfeatures+l], 2);

		if (ans >= min_dist)
			continue
		else {
			index    = i;
			min_dist = ans;
			
		}
	}

	membership[point_id] = index;
}

__kernel void kmeans_swap(__global float  *feature, __global float  *feature_swap,
			int npoints, int nfeatures)
{
	int tid = get_global_id(0);

	for(int i = 0; i <  nfeatures; i++) {
		int li = i * npoints + tid;
		int ri = tid * nfeatures + i;
		feature_swap[li] = feature[ri];
	}

} 
