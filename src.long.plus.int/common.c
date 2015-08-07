#include "common.h"

#ifndef ASSERTION_DEBUG
#define ASSERTION_DEBUG
#endif

char * concatStr(char * s1, char * s2) {
    char *resultStr = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(resultStr, s1);
    strcat(resultStr, s2);
    return resultStr;
}

//#define parseCSV_DEBUG

void parseCSV(char* filename, double** output, int numRows, int numCols) {
    int buffersize = 1024;
    char buf[buffersize];
    // make sure we have buf long enough for reading each line
    assert((sizeof (double) + sizeof (char)) * numCols < buffersize);

    FILE * fstream = fopen(filename, "r");
    assert(fstream != 0);
    *output = (double*) calloc(numRows*numCols, sizeof (double));

    int rowCounter = 0;
    const char * tok;
    int outputCounter = 0;
    while (rowCounter < numRows && fgets(buf, sizeof (buf), fstream)) {
        //        printf("%s", buf);
        // parse a line of CSV
        int rowEleCounter = 0;
        tok = strtok(buf, ",");
        //        printf("%s\n",tok);
        for (; tok&& *tok; tok = strtok(NULL, ",\n")) {
            // remember *output must be parenthesised
            (*output)[outputCounter++] = atof(tok);
            rowEleCounter++;
        }
        
        assert(rowEleCounter == numCols);

        rowCounter++;
    }


    fclose(fstream);
}
#define gerschgorin_v1_DB
//output should be a 2 item array, 
// output[0]=min; output[1]=max;

void gerschgorin_v1(double * output, csrType_local mat) {

    int ierr;
    double local_max = DBL_MIN;
    double local_min = DBL_MAX;

    //[0] is smallest, [1] is largest value
    double global_recvMax, global_recvMin;

    //local max and min search
    int rowIdx;
    int startIdx, endIdx;
    int eleIdx;
    for (rowIdx = 0; rowIdx < mat.num_rows; rowIdx++) {
        startIdx = mat.row_start[rowIdx];
        endIdx = mat.row_start[rowIdx + 1];
        double accumulator = 0.0;
        double diagonalEle = 0.0; // for the case diagonal = 0.0

        for (eleIdx = startIdx; eleIdx < endIdx; eleIdx++) {
            if (rowIdx + mat.start != mat.col_idx[eleIdx]) {
                accumulator += (mat.csrdata[eleIdx] > 0 ? mat.csrdata[eleIdx] : -mat.csrdata[eleIdx]);
            } else {
                diagonalEle = mat.csrdata[eleIdx];
            }
        }

        if (local_max < (accumulator + diagonalEle)) {
            local_max = accumulator + diagonalEle;
        }
        if (local_min > (diagonalEle - accumulator)) {
            local_min = diagonalEle - accumulator;
        }

        //        printf("accumulator: %f, diagobalEle:%f\n", accumulator, diagonalEle);
    }

    //mpi_allreduce mpi_max, mpi_min
    ierr = MPI_Allreduce((void*) &local_min, (void*) &global_recvMin, 1
            , MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    ierr = MPI_Allreduce((void*) &local_max, (void*) &global_recvMax, 1
            , MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    output[0] = global_recvMin;
    output[1] = global_recvMax;

    //    printf ("max:%f, min:%f\n", local_max, local_min);
}
//#define CBCG_chebyshevPolynomialBasisGen_DB_1

void CBCG_chebyshevPolynomialBasisGen(denseType S_mat, csrType_local mat, denseType R, int s
        , double s_alpha, double s_beta, int myid, int numprocs) {
    int ierr;

    // Chebyshev polynomial chunk size
    int cpLocalChunkSize = R.local_num_col * R.local_num_row;
    int S_displ;

    double * AR_global_shape_buffer = (double*) calloc(R.global_num_col * R.global_num_row, sizeof (double));
    //S_mat_dummy which is a transposed S_mat
    // note that the data in S_mat_dummy is actually S_mat tranpose, 
    // so the global(local)_num and global(local)_num are wrong,
    // you need to exchange corresponding value of col and row to use them appropriately
    denseType S_mat_dummy;
    get_same_shape_denseType(S_mat, &S_mat_dummy);
    S_mat_dummy.local_num_col = S_mat.local_num_row;
    S_mat_dummy.local_num_row = S_mat.local_num_col;
    S_mat_dummy.global_num_col = S_mat.global_num_row;
    S_mat_dummy.global_num_row = S_mat.global_num_col;


    // if A is SPD, A*R and A*S(,j) should be of the same shape as R
    denseType spmvBuf;
    get_same_shape_denseType(R, &spmvBuf);

    double s_alpha_by2 = 2.0 * s_alpha;
    double s_beta_by2 = 2.0 * s_beta;



    int sIdx;
    for (sIdx = 0; sIdx < s; sIdx++) {
        if (sIdx == 0) {
            //S(:,1)=r
            dense_entry_copy_disp(R, 0, S_mat_dummy, 0, R.local_num_col * R.local_num_row);
        } else if (sIdx == 1) {
            //S(:,2) = s_alpha * A * r + s_beta * r;
            spmm_csr_v1(mat, R, &spmvBuf, AR_global_shape_buffer, myid, numprocs);
            // res = alpha*mat1 + beta*mat2, TP, third place
            S_displ = cpLocalChunkSize;
            dense_mat_mat_add_TP_targetDisp(spmvBuf, R, S_mat_dummy, S_displ, s_alpha, s_beta, myid, numprocs);
        } else {
            //S(:,sIdx) = s_alpha_by2 * A * S(:,j-1) + s_beta_by2 * S(:,j-1) - S(:,j-2);
            S_displ = (sIdx - 1) * cpLocalChunkSize;
            spmm_csr_info_data_sep_CBCG(mat, R, S_mat_dummy.data, S_displ, &spmvBuf, myid, numprocs);
            dense_array_mat_mat_mat_add_TP_disp(spmvBuf.data, 0
                    , S_mat_dummy.data, (sIdx - 1) * cpLocalChunkSize
                    , S_mat_dummy.data, (sIdx - 2) * cpLocalChunkSize
                    , S_mat_dummy.data, cpLocalChunkSize * sIdx
                    , cpLocalChunkSize
                    , s_alpha_by2, s_beta_by2, -1.0
                    , myid, numprocs);
        }
    }
#ifdef CBCG_chebyshevPolynomialBasisGen_DB_1
    if (myid == numprocs - 3) {
        local_dense_mat_print(S_mat_dummy, myid);
    }
    exit(1);
#endif

    // local transpose, and save to S_mat
    // so, matrix S_mat will be distributed matrix stored by row order
    dense_matrix_local_transpose_row_order(S_mat_dummy, S_mat);

    //
    free(S_mat_dummy.data);
    free(spmvBuf.data);

}
//#define BCBCG_chebyshevPolynomialBasisGen_DB_1

void BCBCG_chebyshevPolynomialBasisGen(denseType S_mat, csrType_local mat, denseType R, int s
        , double s_alpha, double s_beta, int myid, int numprocs) {
    int ierr;

    // Chebyshev polynomial chunk size
    int cpLocalChunkSize = R.local_num_col * R.local_num_row;
    int S_displ;

    double * AR_global_shape_buffer = (double*) calloc(R.global_num_col * R.global_num_row, sizeof (double));
    //S_mat_dummy which is a transposed S_mat
    // note that the data in S_mat_dummy is actually S_mat tranpose, 
    // so the global(local)_num and global(local)_num are wrong,
    // you need to exchange corresponding value of col and row to use them appropriately
    denseType S_mat_dummy;
    get_same_shape_denseType(S_mat, &S_mat_dummy);
    S_mat_dummy.local_num_col = S_mat.local_num_row * R.local_num_col;
    S_mat_dummy.local_num_row = S_mat.local_num_col / R.local_num_col;
#ifdef ASSERTION_DEBUG
    assert(S_mat.local_num_col % R.local_num_col == 0);
    assert(S_mat_dummy.local_num_row == s);
#endif
    S_mat_dummy.global_num_col = S_mat.global_num_row * R.global_num_col;
    S_mat_dummy.global_num_row = S_mat.global_num_col / R.global_num_col;


    // if A is SPD, A*R and A*S(,j) should be of the same shape as R
    denseType spmvBuf;
    get_same_shape_denseType(R, &spmvBuf);

    double s_alpha_by2 = 2.0 * s_alpha;
    double s_beta_by2 = 2.0 * s_beta;



    int sIdx;
    for (sIdx = 0; sIdx < s; sIdx++) {
        if (sIdx == 0) {
            //S(:,1)=r
            dense_entry_copy_disp(R, 0, S_mat_dummy, 0, R.local_num_col * R.local_num_row);
        } else if (sIdx == 1) {
            //S(:,2) = s_alpha * A * r + s_beta * r;
            spmm_csr_v1(mat, R, &spmvBuf, AR_global_shape_buffer, myid, numprocs);
            // res = alpha*mat1 + beta*mat2, TP, third place
            S_displ = cpLocalChunkSize;
            dense_mat_mat_add_TP_targetDisp(spmvBuf, R, S_mat_dummy, S_displ, s_alpha, s_beta, myid, numprocs);
        } else {
            //S(:,sIdx) = s_alpha_by2 * A * S(:,j-1) + s_beta_by2 * S(:,j-1) - S(:,j-2);
            S_displ = (sIdx - 1) * cpLocalChunkSize;
            spmm_csr_info_data_sep_BCBCG(mat, R, S_mat_dummy.data, S_displ, &spmvBuf, myid, numprocs);
            dense_array_mat_mat_mat_add_TP_disp(spmvBuf.data, 0
                    , S_mat_dummy.data, (sIdx - 1) * cpLocalChunkSize
                    , S_mat_dummy.data, (sIdx - 2) * cpLocalChunkSize
                    , S_mat_dummy.data, cpLocalChunkSize * sIdx
                    , cpLocalChunkSize
                    , s_alpha_by2, s_beta_by2, -1.0
                    , myid, numprocs);
        }
    }
#ifdef BCBCG_chebyshevPolynomialBasisGen_DB_1
    if (myid == 0) {
        local_dense_mat_print(S_mat_dummy, myid);
    }
//    exit(1);
#endif

    // local transpose, and save to S_mat
    // so, matrix S_mat will be distributed matrix stored by row order
    dense_matrix_local_transpose_chunk_row_order(S_mat_dummy, S_mat, R.local_num_col);

    //
    free(S_mat_dummy.data);
    free(spmvBuf.data);
    free(AR_global_shape_buffer);

}