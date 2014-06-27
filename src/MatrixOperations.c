#include "MatrixOperations.h"

#define ASSERTION_DEBUG

double* local_dense_colum_2norm(denseType mat) {
    double * norm_array = (double *) calloc(mat.local_num_col, sizeof (double));

    int idx;
    for (idx = 0; idx < mat.local_num_col; idx++)
        norm_array[idx] = 0.0;
    int idx_i, idx_j;
    for (idx_i = 0; idx_i < mat.local_num_row; idx_i++) {
        for (idx_j = 0; idx_j < mat.local_num_col; idx_j++) {
            norm_array[idx_j] += mat.data[idx_i * mat.local_num_col + idx_j]
                    * mat.data[idx_i * mat.local_num_col + idx_j];
        }
    }
    return norm_array;

}

void dense_entry_copy(denseType src, denseType target) {
#ifdef ASSERTION_DEBUG
    assert(src.local_num_col == target.local_num_col);
    assert(src.local_num_row == target.local_num_row);
#endif

    int idx;
    int total_num_ele = src.local_num_col * src.local_num_row;
    for (idx = 0; idx < total_num_ele; idx++) {
        target.data[idx] = src.data[idx];
    }
}

void generate_no_parallel_dense(denseType * mat, int num_row, int num_col) {
    mat->global_num_col = num_col;
    mat->global_num_row = num_row;

    mat->local_num_col = mat->global_num_col;
    mat->local_num_row = mat->global_num_row;

    mat->start_idx = 0;

    mat->data = (double *) calloc(num_row*num_col, sizeof (double));
}

// storage in outMat has been allocated, inMat and outMat have the same size
// inMat is row order matrix, and outMat is the col order matrix
// inMat is a local dense matrix
// outMat is a local dense matrix, all properties of outMat has been set up appropriately

void dense_matrix_local_transpose_row_order(denseType inMat, denseType outMat) {

    int num_col = inMat.local_num_col;
    int num_row = inMat.local_num_row;

    int rowIdx, colIdx;

    for (rowIdx = 0; rowIdx < num_row; rowIdx++) {
        for (colIdx = 0; colIdx < num_col; colIdx++) {
            outMat.data[colIdx * num_row + rowIdx] = inMat.data[rowIdx * num_col + colIdx];
        }
    }
}

// compare to dense_matrix_local_transpose_row_order function,
// the elements of inMat change to vector of size of blocksize, 
// and each of the vectors are stored in row order in memory

void dense_matrix_local_transpose_chunk_row_order(denseType inMat, denseType outMat, int blocksize) {

    int num_col = inMat.local_num_col;
    int num_col_vec = inMat.local_num_col/blocksize;
#ifdef ASSERTION_DEBUG
    assert (inMat.local_num_col%blocksize == 0);
#endif
    int num_row = inMat.local_num_row;
    int num_col_output = num_row * blocksize;

    int rowIdx, colIdx;
    int blockIdx;

    for (rowIdx = 0; rowIdx < num_row; rowIdx++) {
        for (colIdx = 0; colIdx < num_col_vec; colIdx++) {
            int chunkIdx_in=rowIdx * num_col + colIdx*blocksize;
            int chunkIdx_out = colIdx * num_col_output + rowIdx*blocksize;
            for (blockIdx = 0; blockIdx < blocksize; blockIdx++) {
                outMat.data[chunkIdx_out+blockIdx] = inMat.data[chunkIdx_in+blockIdx];
            }
        }
    }
}

void local_dense_mat_print(denseType mat, int myid) {
    int rowIdx, colIdx;

    for (rowIdx = 0; rowIdx < mat.local_num_row; rowIdx++) {
        printf("myid: %d, row: %d\n", myid, rowIdx);
        for (colIdx = 0; colIdx < mat.local_num_col; colIdx++) {
            printf("%f", mat.data[rowIdx * mat.local_num_col + colIdx]);
            printf(" ");
        }
        printf("\n");
    }
    printf("function local_dense_mat_print printing done\n");
}

//#define norm2square_dist_denseMat_col_n_DB

void norm2square_dist_denseMat_col_n(denseType mat, double* result, int n, int myid, int numprocs) {
    int ierr;
    double partial_result = .0;
    int c_n = n - 1;
    int rowIdx;
    int col_num_length = mat.local_num_col;
    if (n > col_num_length) {
        exit(0);
    }

    for (rowIdx = 0; rowIdx < mat.local_num_row; rowIdx++) {
        double temp = mat.data[rowIdx * col_num_length + c_n];
        partial_result += temp * temp;
    }
#ifdef norm2square_dist_denseMat_col_n_DB
    if (myid == 0) {
        printf("partial result:%f\n", partial_result);
    }
#endif

    ierr = MPI_Allreduce((void*) &partial_result, (void*) result, 1
            , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#ifdef norm2square_dist_denseMat_col_n_DB
    if (myid == 2) {
        printf("result:%f\n", *result);
    }
#endif

}

void dense_entry_copy_disp(denseType src, int srcDispStart, denseType target, int tarDispStart, int count) {
#ifdef ASSERTION_DEBUG
    int srcTotal = src.local_num_col * src.local_num_row;
    int targetTotal = target.local_num_col * target.local_num_row;

    assert((srcDispStart + count) <= srcTotal);
    assert((tarDispStart + count) <= targetTotal);
#endif

    int srcIdx, tarIdx;
    int srcEndIdx = srcDispStart + count;
    tarIdx = tarDispStart;
    for (srcIdx = srcDispStart; srcIdx < srcEndIdx; srcIdx++) {

        target.data[tarIdx] = src.data[srcIdx];
        tarIdx++;
    }
}