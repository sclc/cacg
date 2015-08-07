
#include "MatGenerator.h"

void ParallelGen_RandomSymetricSparseMatrix(csrType_local * localMat, matInfo * mat_info \
											, long dim
                                        	, int myid, int numprocs)
{
	if ((long)(dim/(long)numprocs) > 100000) 
	{
		printf ("Error: you have more than 100000 rows per process\n");
		exit(0);
	}

}