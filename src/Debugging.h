/* 
 * File:   Debugging.h
 * Author: scl
 *
 * Created on November 27, 2013, 6:51 PM
 */
#include "DataTypes.h"
#include "mpi.h"
#include <stdio.h>
#include <assert.h>


#ifdef	__cplusplus
extern "C" {
#endif

    void check_coo_matrix_print(cooType mat, matInfo *info);
    void check_csr_matrix_print(csrType_local mat);
    void check_csv_array_print(double* array, long rows, long cols, int myid);
    void check_small_dense_mat (denseType mat , int myid, int numprocs);
    void check_equality_dense (denseType mat1, denseType mat2 , int myid, int numprocs);

#ifdef	__cplusplus
}
#endif


