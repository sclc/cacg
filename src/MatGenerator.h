#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include "DataTypes.h"
#include "common.h"


void ParallelGen_RandomSymetricSparseMatrix(csrType_local * localMat, matInfo * mat_info \
											, long dim
                                        	, int myid, int numprocs);