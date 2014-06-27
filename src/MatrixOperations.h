/* 
 * File:   MatrixOperations.h
 * Author: scl
 *
 * Created on December 17, 2013, 7:40 PM
 */
#include <stdio.h>
#include <stdlib.h>
#include "DataTypes.h"
#include <assert.h>
#include <mpi.h>

#ifndef MATRIXOPERATIONS_H
#define	MATRIXOPERATIONS_H

#ifdef	__cplusplus
extern "C" {
#endif

    double * local_dense_colum_2norm (denseType mat);
    void local_dense_mat_print(denseType mat, int myid);
    void dense_entry_copy (denseType src, denseType target);
    void generate_no_parallel_dense (denseType * mat, int num_row, int num_col);
    
    void dense_matrix_local_transpose_row_order(denseType inMat, denseType outMat);
    void dense_matrix_local_transpose_chunk_row_order(denseType inMat, denseType outMat, int blocksize);
    
    void norm2square_dist_denseMat_col_n (denseType mat, double* result,int n, int myid, int numprocs);
    
    //even if both src and target are denseType matrices,
    // we actually copy consecutive element from src.data to target.data
    void dense_entry_copy_disp(denseType src, int srcDispStart,denseType target, int tarDispStart, int count);

#ifdef	__cplusplus
}
#endif

#endif	/* MATRIXOPERATIONS_H */

