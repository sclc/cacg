#include "DataTypes.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "MatrixOperations.h"

void spmm_csr_serial_v1 (csrType_local matL, denseType matRi, denseType resMat);

void spmm_csr_v1 (csrType_local csr_mat, denseType dense_mat, denseType *res_mat, double * global_swap_zone, int myid, int numprocs);
void spmm_csr_v2 (csrType_local csr_mat, denseType dense_mat, denseType *res_mat, int myid, int numprocs);

//double * global_swap_zone is deprecated

void spmm_csr_info_data_sep_CBCG (csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, int dataDisp
        , denseType *res_mat, int myid, int numprocs);
void spmm_csr_info_data_sep_BCBCG (csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, int dataDisp
        , denseType *res_mat, int myid, int numprocs);

// res = alpha*mat1 + beta*mat2, TP, third place
// note: one each processor, mat1 and mat2 should have the same shape
void dense_mat_mat_add_TP(denseType mat1, denseType mat2, denseType output_mat, double alpha, double beta, int myid, int numprocs);

// res[output_mat_disp ...] = alpha*mat1 + beta*mat2, TP, third place
// note: one each processor, mat1 and mat2 should have the same shape
// res matrix get the result, however, a displacement should be promised
void dense_mat_mat_add_TP_targetDisp(denseType mat1, denseType mat2, denseType output_mat, int output_mat_disp
                                    ,double alpha, double beta, int myid, int numprocs);

// res[output_mat_disp ...] = alpha*mat1[disp1...] + beta*mat2[disp2...] + gamma*mat3[disp3...], TP, third place
// length data items will be processed
// note: one each processor, mat1 and mat2 and mat3 should have the same shape
// res matrix get the result, however, a displacement should be promised
void dense_array_mat_mat_mat_add_TP_disp(double* mat1Data, int mat1Disp
                                             , double* mat2Data, int mat2Disp
                                             , double* mat3Data, int mat3Disp
                                             , double* output_mat, int output_mat_disp
                                             , int length
                                             , double alpha, double beta, double gamma
                                             ,int myid, int numprocs);

//res_mat = mat1**t * mat2
void dense_MtM (denseType mat1, denseType mat2, denseType * res_mat, int myid, int numprocs);

// res_mat = (dis_mat1)transpose * local_mat2
// dis_mat1 is a matrix distributed over processes by rows, and is stored by row order
// local_mat2 is a local matrix, and is stored by row order
// allgatherv to PtA to each processes and calculate redundantly on each processes
void distributedMatTransposeLocalMatMul_v1(denseType *res_mat, denseType dis_mat1, double * local_mat2, int myid,int numprocs);

// res_mat = matLtranspose * matR
// matL and matR are matrices distributed over processes by rows
// matL can be considered as matLtranspose stored by col order 
// calculate partial results on each processes and use mpi_allreduce to get sum to all processes 
void distributedMatLTransposeMatRMul(denseType *resMat, denseType matL, denseType matR, int myid, int numprocs);
void distributedMatLTransposeMatRMul_updated(denseType *resMat, denseType matL, denseType matR, int myid, int numprocs);


// mat**T * mat
// inner product calculation
// mat is a local part of 
// use mpi_allreduce to sum partial results
// resMat has the same copy on each processes
void MatTranposeMatMul (denseType *resMat, denseType mat, int myid, int numprocs);

//matL * matR
void dense_distributedMatL_localMatR_Mul_v1 (denseType * res_mat, denseType matDistL, denseType matLocalR,int myid, int numprocs);