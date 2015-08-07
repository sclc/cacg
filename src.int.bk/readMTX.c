//
#include "readMTX.h"

void readMtx_coo(char* path, char* name, cooType mtr, matInfo info) {

}

void readMtx_info_and_coo(char* path, char* name, matInfo * info, cooType * mat) {
    int ret_code;
    int m, n, nnz;
    MM_typecode matcode;
    FILE *fid;
    int idx;

    fid = fopen(concatStr(path, name), "r");

    if (fid == NULL) {
        printf("Unable to open file %s\n", concatStr(path, name));
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)) {
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }
    ////////////////////////////////////////

    if ((ret_code = mm_read_mtx_crd_size(fid, &m, &n, &nnz)) != 0)
        exit(1);

    info->num_rows = m;
    info->num_cols = n;
    info->nnz = nnz;

    mat->rowIdx = (int *) calloc(info->nnz, sizeof (int));
    mat->colIdx = (int *) calloc(info->nnz, sizeof (int));
    mat->coodata = (double *) calloc(info->nnz, sizeof (double));

    printf("Reading sparse matrix from file ( %s ):\n", concatStr(path, name));

    if (mm_is_pattern(matcode)) {
        // pattern matrix defines sparsity pattern, but not values
        for (idx = 0; idx < info->nnz; idx++) {
            assert(fscanf(fid, " %d %d \n", &(mat->rowIdx[idx]), &(mat->colIdx[idx])) == 2);
            mat->rowIdx[idx]--; //adjust from 1-based to 0-based indexing
            mat->colIdx[idx]--;
            mat->coodata[idx] = 1.0; //use value 1.0 for all nonzero entries 
        }
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for (idx = 0; idx < info->nnz; idx++) {
            int I, J;
            double V; // always read in a double and convert later if necessary

            assert(fscanf(fid, " %d %d %lf \n", &I, &J, &V) == 3);

            mat->rowIdx[idx] = I - 1;
            mat->colIdx[idx] = J - 1;
            mat->coodata[idx] = V;
        }
    } else {
        printf("Unrecognized data type\n");
        exit(1);
    }


    fclose(fid);
    printf(" done reading\n");

    if (mm_is_symmetric(matcode)) { //duplicate off diagonal entries
        int off_diagonals = 0;
        // for spmv feature
        for (idx = 0; idx < info->nnz; idx++) {
            if (mat->rowIdx[idx] != mat->colIdx[idx])
                off_diagonals++;
        }

        int true_nonzeros = 2 * off_diagonals + (info->nnz - off_diagonals);

        int* new_I = (int *) calloc(true_nonzeros, sizeof (int)); 
        int* new_J = (int *) calloc(true_nonzeros, sizeof (int)); 
        double* new_V = (double *) calloc(true_nonzeros, sizeof (double)); 

        int ptr = 0;
        for (idx = 0; idx < info->nnz; idx++) {
            if (mat->rowIdx[idx] != mat->colIdx[idx]) {
                new_I[ptr] = mat->rowIdx[idx];
                new_J[ptr] = mat->colIdx[idx];
                new_V[ptr] = mat->coodata[idx];
                ptr++;
                new_J[ptr] = mat->rowIdx[idx];
                new_I[ptr] = mat->colIdx[idx];
                new_V[ptr] = mat->coodata[idx];
                ptr++;
            } else {
                new_I[ptr] = mat->rowIdx[idx];
                new_J[ptr] = mat->colIdx[idx];
                new_V[ptr] = mat->coodata[idx];
                ptr++;
            }
        }
        assert (ptr == true_nonzeros);
        free(mat->rowIdx);
        free(mat->colIdx);
        free(mat->coodata);

        mat->rowIdx = new_I;
        mat->colIdx = new_J;
        mat->coodata = new_V;
        info->nnz = true_nonzeros;
    } //end symmetric case

}
