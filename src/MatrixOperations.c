#include "MatrixOperations.h"

#define ASSERTION_DEBUG

double* local_dense_colum_2norm(denseType mat) {
    double * norm_array = (double *) calloc(mat.local_num_col, sizeof (double));

    long idx;
    for (idx = 0; idx < mat.local_num_col; idx++)
        norm_array[idx] = 0.0;
    long idx_i, idx_j;
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

    long idx;
    long total_num_ele = src.local_num_col * src.local_num_row;
    for (idx = 0; idx < total_num_ele; idx++) {
        target.data[idx] = src.data[idx];
    }
}

void generate_no_parallel_dense(denseType * mat, long num_row, long num_col) {
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

    long num_col = inMat.local_num_col;
    long num_row = inMat.local_num_row;

    long rowIdx, colIdx;

    for (rowIdx = 0; rowIdx < num_row; rowIdx++) {
        for (colIdx = 0; colIdx < num_col; colIdx++) {
            outMat.data[colIdx * num_row + rowIdx] = inMat.data[rowIdx * num_col + colIdx];
        }
    }
}

// compare to dense_matrix_local_transpose_row_order function,
// the elements of inMat change to vector of size of blocksize, 
// and each of the vectors are stored in row order in memory

void dense_matrix_local_transpose_chunk_row_order(denseType inMat, denseType outMat, long blocksize) {

    long num_col = inMat.local_num_col;
    long num_col_vec = inMat.local_num_col/blocksize;
#ifdef ASSERTION_DEBUG
    assert (inMat.local_num_col%blocksize == 0);
#endif
    long num_row = inMat.local_num_row;
    long num_col_output = num_row * blocksize;

    long rowIdx, colIdx;
    long blockIdx;

    for (rowIdx = 0; rowIdx < num_row; rowIdx++) {
        for (colIdx = 0; colIdx < num_col_vec; colIdx++) {
            long chunkIdx_in=rowIdx * num_col + colIdx*blocksize;
            long chunkIdx_out = colIdx * num_col_output + rowIdx*blocksize;
            for (blockIdx = 0; blockIdx < blocksize; blockIdx++) {
                outMat.data[chunkIdx_out+blockIdx] = inMat.data[chunkIdx_in+blockIdx];
            }
        }
    }
}

void local_dense_mat_print(denseType mat, int myid) {
    long rowIdx, colIdx;

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

void norm2square_dist_denseMat_col_n(denseType mat, double* result, long n, int myid, int numprocs) {
    long ierr;
    double partial_result = .0;
    long c_n = n - 1;
    long rowIdx;
    long col_num_length = mat.local_num_col;
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
    
// int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
//                   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

    ierr = MPI_Allreduce((void*) &partial_result, (void*) result, 1
            , MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#ifdef norm2square_dist_denseMat_col_n_DB
    if (myid == 2) {
        printf("result:%f\n", *result);
    }
#endif

}

void dense_entry_copy_disp(denseType src, long srcDispStart, denseType target, long tarDispStart, long count) {
#ifdef ASSERTION_DEBUG
    long srcTotal = src.local_num_col * src.local_num_row;
    long targetTotal = target.local_num_col * target.local_num_row;

    assert((srcDispStart + count) <= srcTotal);
    assert((tarDispStart + count) <= targetTotal);
#endif

    long srcIdx, tarIdx;
    long srcEndIdx = srcDispStart + count;
    tarIdx = tarDispStart;
    for (srcIdx = srcDispStart; srcIdx < srcEndIdx; srcIdx++) {

        target.data[tarIdx] = src.data[srcIdx];
        tarIdx++;
    }
}
