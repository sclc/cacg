#include "BLAS3.h"
//#define SPMM_COMM_DEBUG
//#define SPMM_CAL_DEBUG_1
//#define SPMM_CAL_DEBUG_2
//#define SPMM_CAL_DEBUG_RES_MAT
//#define distributedMatTransposeLocalMatMul_DEBUG

//#define dense_mat_mat_add_TP_DEBUG

void spmm_csr_serial_v1(csrType_local matL, denseType matR, denseType resMat) {
    // ToDo
}

// in this version of spmm_csr_v1 function, res_mat and dense_mat has the same shape and distribution pattern
//, for I am handling sparse SPD matrix

void spmm_csr_v1(csrType_local csr_mat, denseType dense_mat, denseType *res_mat, double * global_swap_zone, int myid, int numprocs) {
    int ierr;
    int idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    int local_num_row_normal = dense_mat.global_num_row / numprocs;
    int local_num_col_normal = dense_mat.global_num_col;
    int normal_num_elements = local_num_row_normal * local_num_col_normal;

    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, \
                          sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = normal_num_elements;
        displs[idx] = idx * normal_num_elements;

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (dense_mat.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal;
        }
    }
    //    ierr = MPI_Barrier(MPI_COMM_WORLD);
    ierr = MPI_Allgatherv((void *) dense_mat.data, dense_mat.local_num_col * dense_mat.local_num_row, MPI_DOUBLE
            , global_swap_zone, recv_count, displs
            , MPI_DOUBLE, MPI_COMM_WORLD);

#ifdef SPMM_COMM_DEBUG
    int testid = 100;
    if (myid == testid) {
        int global_num_elements = dense_mat.global_num_col * dense_mat.global_num_row;
        printf("myid=%d, ", myid);
        for (idx = 0; idx < global_num_elements + 2; idx++)
            printf("%f, ", global_swap_zone[idx]);
        printf("myid=%d end\n", myid);
    }
#endif
    // spmm using csr format
#ifdef SPMM_CAL_DEBUG_RES_MAT
    int res_debug_idx;
    int res_num_total_ele = res_mat->local_num_row * res_mat->local_num_col;
    for (res_debug_idx = 0; res_debug_idx < res_num_total_ele; res_debug_idx++) {
        if (res_mat->data[res_debug_idx] != 0.0) {
            printf("in spmm_csr_v1, %dth res_mat data inappropriate\n", res_debug_idx);
            exit(1);
        }
    }
    printf("in spmm_csr_v1, res_mat init value is fine\n");
#endif
    int idx_row;
#ifdef SPMM_CAL_DEBUG_2
    printf("in BLAS3.c, myid=%d,number of row: %d\n", myid, csr_mat.num_rows);
#endif
    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        int row_start_idx = csr_mat.row_start[idx_row];
        int row_end_idx = csr_mat.row_start[idx_row + 1];

        int idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            int col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];
#ifdef SPMM_CAL_DEBUG_2
            if (myid == 0 && idx_row == 0) {
                printf("in BLAS3.c, csr_data=%f\n", csr_data);
            }
#endif
            int block_size = dense_mat.local_num_col;
            int block_idx;
            for (block_idx = 0; block_idx < block_size; block_idx++) {
#ifdef SPMM_CAL_DEBUG_2
                if (myid == 0 && idx_row == 0 && block_idx == 0) {
                    printf("in BLAS3.c, before cumulation, res_data=%f\n", res_buffer[0]);
                    printf("in BLAS3.c, before cumulation, RHS_data=%f\n", global_swap_zone[col_idx * dense_mat.local_num_col + block_idx]);
                    printf("in BLAS3.c, before cumulation, col_idx=%d\n", col_idx);
                    printf("in BLAS3.c, before cumulation, dense_mat.local_num_col=%d\n", dense_mat.local_num_col);

                }
#endif
                res_buffer[idx_row * res_mat->local_num_col + block_idx] +=
                        csr_data * global_swap_zone[col_idx * dense_mat.local_num_col + block_idx];
#ifdef SPMM_CAL_DEBUG_2
                if (myid == 0 && idx_row == 0 && block_idx == 0) {
                    printf("in BLAS3.c, after cumulation, res_data=%f\n\n", res_buffer[0]);
                }
#endif
            }
        }
    }
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;

#ifdef SPMM_CAL_DEBUG_1
    double * local_norm = local_dense_colum_2norm(*res_mat);
    double * global_norm = (double *) calloc(res_mat->global_num_col, sizeof (double));
    int temp_idx;
    printf("in BLAS3.c, myid=%d ", myid);
    for (temp_idx = 0; temp_idx < res_mat->local_num_col; temp_idx++) {
        printf("%f, ", local_norm[temp_idx]);
    }
    printf("local norm ending \n");


    ierr = MPI_Reduce((void*) local_norm, (void*) global_norm, res_mat->global_num_col
            , MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {

        printf("global squared norm array: \n");
        for (temp_idx = 0; temp_idx < res_mat->global_num_col; temp_idx++) {
            printf("%f \n", global_norm[temp_idx]);
        }
        printf("norm ending\n");
    }
    free(global_norm);
    free(local_norm);
#endif
    /////////
#ifdef SPMM_CAL_DEBUG_2
    int spmm_cal_debug_2_i = 0;
    int spmm_cal_debug_2_j = 0;

    printf("in BLAS3.c, myid=%d,\t\t", myid);
    printf("%f\n", res_mat->data[spmm_cal_debug_2_i * res_mat->global_num_col + spmm_cal_debug_2_j]);
#endif

}
//
void spmm_csr_v2(csrType_local csr_mat, denseType dense_mat, denseType *res_mat, int myid, int numprocs) {
    int ierr;
    int idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    int local_num_row_normal = dense_mat.global_num_row / numprocs;
    int local_num_col_normal = dense_mat.global_num_col;
    int normal_num_elements = local_num_row_normal * local_num_col_normal;

    double *recv_buffer = (double*)calloc(dense_mat.global_num_col * dense_mat.global_num_row, sizeof(double));
    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = normal_num_elements;
        displs[idx] = idx * normal_num_elements;

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (dense_mat.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal;
        }
    }
    //    ierr = MPI_Barrier(MPI_COMM_WORLD);
    ierr = MPI_Allgatherv((void *) dense_mat.data, dense_mat.local_num_col * dense_mat.local_num_row, MPI_DOUBLE
            , recv_buffer, recv_count, displs
            , MPI_DOUBLE, MPI_COMM_WORLD);

    // spmm using csr format
    int idx_row;
#ifdef SPMM_CAL_DEBUG_2
    printf("in BLAS3.c, myid=%d,number of row: %d\n", myid, csr_mat.num_rows);
#endif
    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        int row_start_idx = csr_mat.row_start[idx_row];
        int row_end_idx = csr_mat.row_start[idx_row + 1];

        int idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            int col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];
            int block_size = dense_mat.local_num_col;
            int block_idx;
            for (block_idx = 0; block_idx < block_size; block_idx++) {
                res_buffer[idx_row * res_mat->local_num_col + block_idx] +=
                        csr_data * recv_buffer[col_idx * dense_mat.local_num_col + block_idx];
            }
        }
    }
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;
    
    free(recv_buffer);

}
// res = alpha*mat1 + beta*mat2, TP, third place
// note: one each processor, mat1 and mat2 should have the same shape

void dense_mat_mat_add_TP(denseType mat1, denseType mat2, denseType output_mat, \
    double alpha, double beta, int myid, int numprocs) {
    
    int num_rows = mat1.local_num_row;
    int num_cols = mat1.local_num_col;

#ifdef dense_mat_mat_add_TP_DEBUG    
    assert(mat1.local_num_row == mat2.local_num_row);
    assert(mat1.local_num_col == mat2.local_num_col);
    assert(mat1.local_num_col == output_mat.local_num_col);
    assert(mat1.local_num_row == output_mat.local_num_row);
#endif
    int idx_i;
    int idx_j;
    for (idx_i = 0; idx_i < num_rows; idx_i++) {
        for (idx_j = 0; idx_j < num_cols; idx_j++) {
            int mat_idx = idx_i * num_cols + idx_j;
#ifdef dense_mat_mat_add_TP_DEBUG
            if (myid == 0 && (mat_idx == 0 || mat_idx == (num_rows * num_cols - 1))) {
                printf("in dense_mat_mat_add_TP, before calculation, myid=%d, mat1 values: %f; mat2 values: %f; output values: %f\n"
                        , myid, mat1.data[mat_idx], mat2.data[mat_idx], output_mat.data[mat_idx]);
            }
#endif
            output_mat.data[mat_idx] =
                    alpha * mat1.data[mat_idx] + beta * mat2.data[mat_idx];
#ifdef dense_mat_mat_add_TP_DEBUG
            if (myid == 0 && (mat_idx == 0 || mat_idx == (num_rows * num_cols - 1))) {
                printf("in dense_mat_mat_add_TP, before calculation, myid=%d, mat1 values: %f; mat2 values: %f; output values: %f\n"
                        , myid, mat1.data[mat_idx], mat2.data[mat_idx], output_mat.data[mat_idx]);
            }
#endif            
        }
    }
}

void dense_MtM(denseType mat1, denseType mat2, denseType * res_mat, int myid, int numprocs) {

}

void distributedMatTransposeLocalMatMul_v1(denseType *res_mat, denseType dis_mat1, double * local_mat2, int myid, int numprocs) {

    int ierr;

    //since A is SPD, Pt_A = (A_P)^t
    int global_mat1_transposed_col_order_row_num = dis_mat1.global_num_col;
    int global_mat1_transposed_col_order_col_num = dis_mat1.global_num_row;

    double * global_mat1_transposed_col_order = (double *) calloc(dis_mat1.global_num_col * dis_mat1.global_num_row, sizeof (double));

    double * res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, sizeof (double));

    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    int local_num_row_normal = dis_mat1.global_num_row / numprocs;
    int local_num_col_normal = dis_mat1.global_num_col;
    int normal_num_elements = local_num_row_normal * local_num_col_normal;

    int idx;
#ifdef distributedMatTransposeLocalMatMul_DEBUG
    int global_ele_counter = 0;
#endif
    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = normal_num_elements;
        displs[idx] = idx * normal_num_elements;

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (dis_mat1.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal;
        }
#ifdef distributedMatTransposeLocalMatMul_DEBUG
        global_ele_counter += recv_count[idx];
#endif
    }
#ifdef distributedMatTransposeLocalMatMul_DEBUG
    assert(global_ele_counter == (dis_mat1.global_num_col * dis_mat1.global_num_row));
#endif


    ierr = MPI_Allgatherv((void *) dis_mat1.data, dis_mat1.local_num_col * dis_mat1.local_num_row, MPI_DOUBLE
            , global_mat1_transposed_col_order, recv_count, displs
            , MPI_DOUBLE, MPI_COMM_WORLD);
#ifdef distributedMatTransposeLocalMatMul_DEBUG
    int debug_row_idx = 0;
    int debug_col_idx = 47;
    //        printf ("MPI_Allgatherv test; row:%d, col:%d, value:%f\n", debug_row_idx, debug_col_idx, 
    //                global_mat1_transposed_col_order[debug_col_idx * global_mat1_transposed_col_order_row_num + debug_row_idx]);
    //        printf ("aquired ele: %d\n",debug_col_idx * global_mat1_transposed_col_order_row_num + debug_row_idx);
    debug_row_idx = 31;
    debug_col_idx = 0;
    printf("buffer value test; row:%d, col:%d, val:%f\n", debug_row_idx,
            debug_col_idx, local_mat2[debug_row_idx * res_mat->local_num_col + debug_col_idx]);
#endif
    int rowIdx, colIdx, kIdx;
    for (kIdx = 0; kIdx < global_mat1_transposed_col_order_col_num; kIdx++) {
        for (rowIdx = 0; rowIdx < res_mat->local_num_row; rowIdx++) {
            double temp = global_mat1_transposed_col_order[kIdx * global_mat1_transposed_col_order_row_num + rowIdx];
#ifdef distributedMatTransposeLocalMatMul_DEBUG
            printf("myid:%d, temp: %f\n", myid, temp);
#endif
            for (colIdx = 0; colIdx < res_mat->local_num_col; colIdx++) {
                res_buffer[rowIdx * res_mat->local_num_col + colIdx] += local_mat2[kIdx * res_mat->local_num_col + colIdx]
                        * temp;
            }
        }
    }

    free(res_mat->data);
    res_mat->data = res_buffer;

    free(global_mat1_transposed_col_order);
}

// matL' * matR
// matL and matR should be of the same shape
void distributedMatLTransposeMatRMul(denseType *resMat, denseType matL, denseType matR, int myid, int numprocs) {

    int ierr;
    double * res_local_buffer = (double *) calloc(resMat->local_num_col * resMat->local_num_row, sizeof (double));

    int rowIdx, colIdx, kIdx;
    int mat_row_num = matL.local_num_col;
    int mat_col_num = matL.local_num_row;
    int res_square_col_row_num = mat_row_num;

    for (kIdx = 0; kIdx < mat_col_num; kIdx++) {
        for (rowIdx = 0; rowIdx < res_square_col_row_num; rowIdx++) {
            double temp = matL.data[kIdx * mat_row_num + rowIdx];
            for (colIdx = 0; colIdx < res_square_col_row_num; colIdx++) {
                res_local_buffer[rowIdx * res_square_col_row_num + colIdx] += matR.data[kIdx * matR.local_num_col + colIdx]
                        * temp;
            }
        }
    }

    // to sum partial results using MPI_Allreduce
    int send_count = resMat->local_num_col * resMat->local_num_row;
    ierr = MPI_Allreduce((void*) res_local_buffer, (void*) (resMat->data), send_count
            , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(res_local_buffer);

}
// matL' * matR
// matL and matR can be of different shape, 
// however, the #row of matL and #row of matR must be identical
void distributedMatLTransposeMatRMul_updated(denseType *resMat, denseType matL, denseType matR, int myid, int numprocs) {

    int ierr;
    double * res_local_buffer = (double *) calloc(resMat->local_num_col * resMat->local_num_row, sizeof (double));

    int rowIdx, colIdx, kIdx;
    int matTrans_row_num = matL.local_num_col;
    int common_mat_col_num = matL.local_num_row;
    int res_row_num = matL.global_num_col;
    int res_col_num = matR.global_num_col;

    for (kIdx = 0; kIdx < common_mat_col_num; kIdx++) {
        for (rowIdx = 0; rowIdx < res_row_num; rowIdx++) {
            double temp = matL.data[kIdx * matTrans_row_num + rowIdx];
            for (colIdx = 0; colIdx < res_col_num; colIdx++) {
                res_local_buffer[rowIdx * res_col_num + colIdx] += matR.data[kIdx * matR.local_num_col + colIdx]
                        * temp;
            }
        }
    }

    // to sum partial results using MPI_Allreduce
    int send_count = resMat->local_num_col * resMat->local_num_row;
    ierr = MPI_Allreduce((void*) res_local_buffer, (void*) (resMat->data), send_count
            , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(res_local_buffer);

}

void MatTranposeMatMul(denseType *resMat, denseType mat, int myid, int numprocs) {

    int ierr;
    double * res_local_buffer = (double *) calloc(resMat->local_num_col * resMat->local_num_row, sizeof (double));

    if (resMat->data == 0) {
        resMat->data = (double*) calloc(resMat->local_num_col * resMat->local_num_row, sizeof (double));
    }

    int rowIdx, colIdx, kIdx;
    int mat_row_num = mat.local_num_col;
    int mat_col_num = mat.local_num_row;
    int res_square_col_row_num = mat_row_num;

    for (kIdx = 0; kIdx < mat_col_num; kIdx++) {
        for (rowIdx = 0; rowIdx < res_square_col_row_num; rowIdx++) {
            double temp = mat.data[kIdx * mat_row_num + rowIdx];
            for (colIdx = 0; colIdx < res_square_col_row_num; colIdx++) {
                res_local_buffer[rowIdx * res_square_col_row_num + colIdx] += mat.data[kIdx * mat.local_num_col + colIdx]
                        * temp;
            }
        }
    }

    // to sum partial results using MPI_Allreduce
    int send_count = resMat->local_num_col * resMat->local_num_row;
    ierr = MPI_Allreduce((void*) res_local_buffer, (void*) (resMat->data), send_count
            , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    free(res_local_buffer);
}

#define dense_distributedMatL_localMatR_Mul_v1_ASSERTION

void dense_distributedMatL_localMatR_Mul_v1(denseType * res_mat, denseType matDistL, denseType matLocalR, int myid, int numprocs) {

    int row_num = res_mat->local_num_row;
    int col_num = res_mat->local_num_col;
    int k_num = matDistL.local_num_col;

    double *res_buf = (double*) calloc(row_num*col_num, sizeof (double));
#ifdef dense_distributedMatL_localMatR_Mul_v1_ASSERTION
    assert(res_mat->local_num_row == matDistL.local_num_row);
    assert(res_mat->local_num_col == matLocalR.local_num_col);
    assert(matDistL.local_num_col == matLocalR.local_num_row);

#endif
    int rowIdx;
    int colIdx;
    int kIdx;
    double temp;
    for (rowIdx = 0; rowIdx < row_num; rowIdx++) {
        for (kIdx = 0; kIdx < k_num; kIdx++) {
            temp = matDistL.data[rowIdx * k_num + kIdx];
            for (colIdx = 0; colIdx < col_num; colIdx++) {
                res_buf[rowIdx * col_num + colIdx] += matLocalR.data[kIdx * col_num + colIdx]
                        * temp;
            }
        }
    }

    if (res_mat->data != 0) {
        free(res_mat->data);
    }
    res_mat->data = res_buf;
}
#define dense_mat_mat_add_TP_targetDisp_ASSERTION

void dense_mat_mat_add_TP_targetDisp(denseType mat1, denseType mat2, denseType output_mat, int output_mat_disp
        , double alpha, double beta, int myid, int numprocs) {

    int num_rows = mat1.local_num_row;
    int num_cols = mat1.local_num_col;


    int idx_i;
    int idx_j;
    for (idx_i = 0; idx_i < num_rows; idx_i++) {
        for (idx_j = 0; idx_j < num_cols; idx_j++) {
            int mat_idx = idx_i * num_cols + idx_j;
#ifdef dense_mat_mat_add_TP_targetDisp_ASSERTION
            assert((output_mat_disp + mat_idx)< (output_mat.local_num_col * output_mat.local_num_row));
#endif
            output_mat.data[ output_mat_disp + mat_idx] =
                    alpha * mat1.data[mat_idx] + beta * mat2.data[mat_idx];
        }
    }
}

// requirement: the changing elements in dataSrc should be allocated consecutively
// actually, this is a SpMV calculation
void spmm_csr_info_data_sep_CBCG(csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, int dataDisp
        , denseType *res_mat, int myid, int numprocs) {

    int ierr;
    int idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    int local_num_row_normal = dense_mat_info.global_num_row / numprocs;
    int local_num_col_normal = 1;
    int normal_num_elements = local_num_row_normal * local_num_col_normal;

    // recvBuf
    double * recvBuf = (double*)calloc( dense_mat_info.global_num_col * dense_mat_info.global_num_row, sizeof(double));
    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = normal_num_elements;
        displs[idx] = idx * normal_num_elements;

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (dense_mat_info.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal;
        }
    }
 
    ierr = MPI_Allgatherv((void *) (dataSrc+dataDisp), dense_mat_info.local_num_col * dense_mat_info.local_num_row, MPI_DOUBLE
            , recvBuf, recv_count, displs
            , MPI_DOUBLE, MPI_COMM_WORLD);


    // spmv using csr format
    int idx_row;
    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        int row_start_idx = csr_mat.row_start[idx_row];
        int row_end_idx = csr_mat.row_start[idx_row + 1];

        int idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            int col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];

            res_buffer[idx_row] += csr_data * recvBuf[col_idx];
        }
    }
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;

}

//
void spmm_csr_info_data_sep_BCBCG(csrType_local csr_mat, denseType dense_mat_info, double * dataSrc, int dataDisp
        , denseType *res_mat, int myid, int numprocs) {

    int ierr;
    int idx;
    // gather all data from all processes
    int recv_count[numprocs];
    int displs[numprocs];

    int local_num_row_normal = dense_mat_info.global_num_row / numprocs;
    int local_num_col_normal = dense_mat_info.local_num_col;
    int normal_num_elements = local_num_row_normal * local_num_col_normal;

    // recvBuf
    double * recvBuf = (double*)calloc( dense_mat_info.global_num_col * dense_mat_info.global_num_row, sizeof(double));
    // values allocated by calloc() is initialized to zero
    double *res_buffer = (double *) calloc(res_mat->local_num_col * res_mat->local_num_row, sizeof (double));

    for (idx = 0; idx < numprocs; idx++) {
        recv_count[idx] = normal_num_elements;
        displs[idx] = idx * normal_num_elements;

        if (idx == (numprocs - 1)) {
            recv_count[idx] = (dense_mat_info.global_num_row - local_num_row_normal * (numprocs - 1))
                    * local_num_col_normal;
        }
    }
 
    ierr = MPI_Allgatherv((void *) (dataSrc+dataDisp), dense_mat_info.local_num_col * dense_mat_info.local_num_row, MPI_DOUBLE
            , recvBuf, recv_count, displs
            , MPI_DOUBLE, MPI_COMM_WORLD);


    // spmv using csr format
    int idx_row;
    for (idx_row = 0; idx_row < csr_mat.num_rows; idx_row++) {
        int row_start_idx = csr_mat.row_start[idx_row];
        int row_end_idx = csr_mat.row_start[idx_row + 1];

        int idx_data;
        for (idx_data = row_start_idx; idx_data < row_end_idx; idx_data++) {
            int col_idx = csr_mat.col_idx[idx_data];
            double csr_data = csr_mat.csrdata[idx_data];
            int block_size = dense_mat_info.global_num_col;
            int block_idx;
            for (block_idx = 0; block_idx < block_size; block_idx++) {
                res_buffer[idx_row * res_mat->local_num_col + block_idx] +=
                        csr_data * recvBuf[col_idx * dense_mat_info.global_num_col + block_idx];
            }
        }
    }
    // Data zone changes
    if (res_mat->data != 0) {
        free(res_mat->data);
    } else {
        exit(0);
    }
    res_mat->data = res_buffer;

}

void dense_array_mat_mat_mat_add_TP_disp(double* mat1Data, int mat1Disp
                                       , double* mat2Data, int mat2Disp
                                       , double* mat3Data, int mat3Disp
                                       , double* output_mat, int output_mat_disp
                                       , int length
                                       , double alpha, double beta, double gamma
                                       ,int myid, int numprocs){
    
    int idx;
    
    for (idx=0; idx<length;idx++){
        output_mat[output_mat_disp+idx] = alpha * mat1Data[mat1Disp+idx] 
                                        + beta  * mat2Data[mat2Disp+idx]
                                        + gamma * mat3Data[mat3Disp+idx];
    }
    
}