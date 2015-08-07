/* 
 * File:   SparseMatrixDistribution.h
 * Author: scl
 *
 * Created on December 1, 2013, 7:27 PM
 */

#include <mpi.h>
#include "common.h"
#include "readMTX.h"

#ifdef	__cplusplus
extern "C" {
#endif

    void Sparse_Csr_Matrix_Distribution(csrType_local * localMat, matInfo * mat_info,int myid, int numprocs
                                           , char* path, char* mtx_filename);


#ifdef	__cplusplus
}
#endif


