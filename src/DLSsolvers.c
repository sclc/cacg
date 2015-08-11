#include "DLSsolvers.h"

// lhsMat * resMat = rhsMat; AX=B 
//#define LUsolver_v1_DB
// lhsMat values will be changed
void LUsolver_v1(denseType * resMat, denseType lhsMat, denseType rhsMat, int myid, int numprocs) {

    double * Y_buffer = (double*) calloc(rhsMat.local_num_col * rhsMat.local_num_row, sizeof (double));
    long rhs_col_num = rhsMat.local_num_col;

    // Crout's LU decomposition, Crout's algorithm
    // u(i,i) = 1
    long idx;
    long lhs_row_num = lhsMat.local_num_row;
    long lhs_col_num = lhsMat.local_num_col;
    double temp_update_u;
    double temp_update_partial;

    // actually, here we need lhs_col_num = lhs_row_num
    for (idx = 0; idx < lhs_col_num; idx++) {
        temp_update_u = 1.0 / lhsMat.data[idx * lhs_col_num + idx];
        long colIdx_u;
        colIdx_u = idx + 1;
        // calculate upper triangular element
        // to get L(rowIdx,colIdx)
        for (; colIdx_u < lhs_col_num; colIdx_u++) {
            lhsMat.data[idx * lhs_col_num + colIdx_u] *= temp_update_u;
        }
        // calculate lower triangular element
        long rowIdx_lu = idx + 1;
        for (; rowIdx_lu < lhs_row_num; rowIdx_lu++) {
            temp_update_partial = lhsMat.data[rowIdx_lu * lhs_col_num + idx];
            long colIdx_lu = idx + 1;
            for (; colIdx_lu < lhs_col_num; colIdx_lu++) {
                lhsMat.data[rowIdx_lu * lhs_col_num + colIdx_lu] = lhsMat.data[rowIdx_lu * lhs_col_num + colIdx_lu]
                        - temp_update_partial
                        * lhsMat.data[idx * lhs_col_num + colIdx_lu];
            }
        }
    }
#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        //        check_csv_array_print(lhsMat.data, lhsMat.local_num_row, lhsMat.local_num_col, myid);
    }
#endif

    // forward substitution
    long fs_i, fs_j, fs_k;
    double temp_l1, temp_l2;

    for (fs_i = 0; fs_i < lhs_row_num; fs_i++) {
        for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
            Y_buffer[fs_i * rhs_col_num + fs_j] = rhsMat.data[fs_i * rhs_col_num + fs_j];
        }
        //note: here we cannot use fs_k<fs_i-1, otherwise, bug happens 
        for (fs_k = 0; fs_k < fs_i; fs_k++) {
            temp_l1 = lhsMat.data[fs_i * lhs_col_num + fs_k ];
            for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
                Y_buffer[fs_i * rhs_col_num + fs_j] -= Y_buffer[fs_k * rhs_col_num + fs_j] * temp_l1;
            }
        }

        temp_l2 = 1.0 / lhsMat.data[fs_i * lhs_col_num + fs_i ];
        for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
            Y_buffer[fs_i * rhs_col_num + fs_j] *= temp_l2;
        }
    }

#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        // check_csv_array_print(Y_buffer, rhsMat.local_num_row, rhsMat.local_num_col, myid);
    }
#endif

    // backward substitution
    long bs_i, bs_j, bs_k;
    double temp_bs;

    for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
        resMat->data[(lhs_row_num - 1) * rhs_col_num + bs_j] = Y_buffer[(lhs_row_num - 1) * rhs_col_num + bs_j];
    }

    for (bs_i = (lhs_row_num - 2); bs_i >= 0; bs_i--) {
        for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
            resMat->data[bs_i * rhs_col_num + bs_j] = Y_buffer[bs_i * rhs_col_num + bs_j];
        }

        for (bs_k = (bs_i + 1); bs_k < lhs_col_num; bs_k++) {
            temp_bs = lhsMat.data[bs_i * lhs_col_num + bs_k];
            for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
                resMat->data[bs_i * rhs_col_num + bs_j] -= resMat->data[bs_k * rhs_col_num + bs_j] * temp_bs;
            }
        }
    }
#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        check_csv_array_print(resMat->data, resMat->local_num_row, resMat->local_num_col, myid);
    }
#endif

    free(Y_buffer);
}

// lhsMat * resMat = rhsMat; AX=B 
// lhsMat values will not be changed
void LUsolver_v1_KeeplhsMat(denseType * resMat, denseType lhsMatdb, denseType rhsMat, int myid, int numprocs) {

    double * Y_buffer = (double*) calloc(rhsMat.local_num_col * rhsMat.local_num_row, sizeof (double));
    
    denseType lhsMatDummy;
    get_same_shape_denseType(lhsMatdb, &lhsMatDummy);
    dense_entry_copy(lhsMatdb, lhsMatDummy);
    
    long rhs_col_num = rhsMat.local_num_col;

    // Crout's LU decomposition, Crout's algorithm
    // u(i,i) = 1
    long idx;
    long lhs_row_num = lhsMatdb.local_num_row;
    long lhs_col_num = lhsMatdb.local_num_col;
    double temp_update_u;
    double temp_update_partial;

    // actually, here we need lhs_col_num = lhs_row_num
    for (idx = 0; idx < lhs_col_num; idx++) {
        temp_update_u = 1.0 / lhsMatDummy.data[idx * lhs_col_num + idx];
        long colIdx_u;
        colIdx_u = idx + 1;
        // calculate upper triangular element
        // to get L(rowIdx,colIdx)
        for (; colIdx_u < lhs_col_num; colIdx_u++) {
            lhsMatDummy.data[idx * lhs_col_num + colIdx_u] *= temp_update_u;
        }
        // calculate lower triangular element
        long rowIdx_lu = idx + 1;
        for (; rowIdx_lu < lhs_row_num; rowIdx_lu++) {
            temp_update_partial = lhsMatDummy.data[rowIdx_lu * lhs_col_num + idx];
            long colIdx_lu = idx + 1;
            for (; colIdx_lu < lhs_col_num; colIdx_lu++) {
                lhsMatDummy.data[rowIdx_lu * lhs_col_num + colIdx_lu] = lhsMatDummy.data[rowIdx_lu * lhs_col_num + colIdx_lu]
                        - temp_update_partial
                        * lhsMatDummy.data[idx * lhs_col_num + colIdx_lu];
            }
        }
    }
#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        //        check_csv_array_print(lhsMat.data, lhsMat.local_num_row, lhsMat.local_num_col, myid);
    }
#endif

    // forward substitution
    long fs_i, fs_j, fs_k;
    double temp_l1, temp_l2;

    for (fs_i = 0; fs_i < lhs_row_num; fs_i++) {
        for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
            Y_buffer[fs_i * rhs_col_num + fs_j] = rhsMat.data[fs_i * rhs_col_num + fs_j];
        }
        //note: here we cannot use fs_k<fs_i-1, otherwise, bug happens 
        for (fs_k = 0; fs_k < fs_i; fs_k++) {
            temp_l1 = lhsMatDummy.data[fs_i * lhs_col_num + fs_k ];
            for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
                Y_buffer[fs_i * rhs_col_num + fs_j] -= Y_buffer[fs_k * rhs_col_num + fs_j] * temp_l1;
            }
        }

        temp_l2 = 1.0 / lhsMatDummy.data[fs_i * lhs_col_num + fs_i ];
        for (fs_j = 0; fs_j < rhs_col_num; fs_j++) {
            Y_buffer[fs_i * rhs_col_num + fs_j] *= temp_l2;
        }
    }

#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        // check_csv_array_print(Y_buffer, rhsMat.local_num_row, rhsMat.local_num_col, myid);
    }
#endif

    // backward substitution
    long bs_i, bs_j, bs_k;
    double temp_bs;

    for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
        resMat->data[(lhs_row_num - 1) * rhs_col_num + bs_j] = Y_buffer[(lhs_row_num - 1) * rhs_col_num + bs_j];
    }

    for (bs_i = (lhs_row_num - 2); bs_i >= 0; bs_i--) {
        for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
            resMat->data[bs_i * rhs_col_num + bs_j] = Y_buffer[bs_i * rhs_col_num + bs_j];
        }

        for (bs_k = (bs_i + 1); bs_k < lhs_col_num; bs_k++) {
            temp_bs = lhsMatDummy.data[bs_i * lhs_col_num + bs_k];
            for (bs_j = 0; bs_j < rhs_col_num; bs_j++) {
                resMat->data[bs_i * rhs_col_num + bs_j] -= resMat->data[bs_k * rhs_col_num + bs_j] * temp_bs;
            }
        }
    }
#ifdef LUsolver_v1_DB
    if (myid == numprocs - 1) {
        check_csv_array_print(resMat->data, resMat->local_num_row, resMat->local_num_col, myid);
    }
#endif

    free(Y_buffer);
    delete_denseType(lhsMatDummy);
}