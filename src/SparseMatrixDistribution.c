//#define DEBUG_DATA_DIST
//#define DEBUG_DATA_DIST_2
//#define DEBUG_DATA_DIST_LOCAL_ROW_START

#include "SparseMatrixDistribution.h"

void Sparse_Csr_Matrix_Distribution(csrType_local * localMat, matInfo * mat_info, int myid, int numprocs
        , char* path, char* mtx_filename) {
    // use MPI blocking communication
    cooType globalMat;

    csrType_local rank_0_global_csr;

    int * col_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    int * col_local_length = (int *) calloc(numprocs, sizeof (int));

    int * val_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    int * val_local_length = (int *) calloc(numprocs, sizeof (int));

    int rows_per_proc, num_total, num_row_local;

    int idx;
    int ierr;

    int num_row_col[2];
    if (myid == 0) {
        // rank 0 read mtr

        readMtx_info_and_coo(path, mtx_filename, mat_info, & globalMat);
        printf("there are %d elements in matrix \n", mat_info->nnz);
#ifdef DEBUG_DATA_DIST
        //check_coo_matrix_print(globalMat, *mat_info);
#endif
        Converter_Coo2Csr(globalMat, &rank_0_global_csr, mat_info);
#ifdef DEBUG_DATA_DIST
        //check_csr_matrix_print(rank_0_global_csr);
#endif
        num_row_col[0] = mat_info->num_rows;
        num_row_col[1] = mat_info->num_cols;

        delete_cooType(globalMat);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast((void *) num_row_col, 2, MPI_INT, 0, MPI_COMM_WORLD);
    num_total = num_row_col[0];
    rows_per_proc = num_total / numprocs;

    if (myid != 0) {
        rank_0_global_csr.row_start = (int *) calloc(num_total + 1, sizeof (int));
    }
#ifdef DEBUG_DATA_DIST
    printf("myid=%d, num_row_col[0]=%d, num_row_col[1]=%d \n"
            , myid, num_row_col[0], num_row_col[1]);
#endif


    if (myid == (numprocs - 1)) {
        num_row_local = num_total - rows_per_proc * (numprocs - 1);
    } else {
        num_row_local = rows_per_proc;
    }

    localMat->num_rows = num_row_local;

    localMat->num_cols = num_row_col[1];

    localMat->row_start = (int *) calloc(num_row_local + 1, sizeof (int));
    localMat->start = myid * rows_per_proc;
#ifdef DEBUG_DATA_DIST
    //    printf("myid=%d, localMat->start=%d, localMat->num_rows=%d\n", myid, localMat->start, localMat->num_rows);
#endif    

    ierr = MPI_Bcast((void*) rank_0_global_csr.row_start, num_total + 1, MPI_INT
            , 0, MPI_COMM_WORLD);
#ifdef DEBUG_DATA_DIST
    printf("myid=%d, ", myid);
    for (idx = 0; idx < num_total + 1; idx++) {
        printf("%d, ", rank_0_global_csr.row_start[idx]);
    }
    printf("\n");
#endif
    // for local_car is only for local data, we need to shift row_start value
    // , or say, the first value of row_start should be 0 for all processes
    for (idx = 0; idx <= num_row_local; idx++) {
        localMat->row_start[idx] = rank_0_global_csr.row_start[myid * rows_per_proc + idx]
                - rank_0_global_csr.row_start[myid * rows_per_proc];
    }

#ifdef DEBUG_DATA_DIST_LOCAL_ROW_START
    printf("##################local row_start test myid=%d, \t", myid);
    for (idx = 0; idx <= num_row_local; idx++) {
        printf("%d, ", localMat->row_start[idx]);
    }
    printf("\n");
#endif

    for (idx = 0; idx < numprocs; idx++) {
        int row_start = idx * rows_per_proc;
        int row_end = row_start + rows_per_proc;
        if (idx == (numprocs - 1))
            row_end = num_total;

        col_local_start_idx[idx] = rank_0_global_csr.row_start[row_start];
        col_local_length[idx] = rank_0_global_csr.row_start[row_end]
                - rank_0_global_csr.row_start[row_start];

        val_local_start_idx[idx] = rank_0_global_csr.row_start[row_start];
        val_local_length[idx] = rank_0_global_csr.row_start[row_end]
                - rank_0_global_csr.row_start[row_start];
    }
#ifdef DEBUG_DATA_DIST
    int total_entries = 0;
    for (idx = 0; idx < numprocs; idx++)
        total_entries += col_local_length[idx];
    printf("#########myid=%d, total_entries=%d\n", myid, total_entries);
    printf("#########myid=%d, local_entries=%d\n", myid, col_local_length[myid]);

#endif

    localMat->col_idx = (int *) calloc(col_local_length[myid], sizeof (int));
    localMat->csrdata = (double*) calloc(val_local_length[myid], sizeof (double));

    ierr = MPI_Scatterv((void *) rank_0_global_csr.col_idx, col_local_length
            , col_local_start_idx, MPI_INT
            , localMat->col_idx, col_local_length[myid]
            , MPI_INT, 0, MPI_COMM_WORLD);

    ierr = MPI_Scatterv((void *) rank_0_global_csr.csrdata, val_local_length
            , val_local_start_idx, MPI_DOUBLE
            , localMat->csrdata, val_local_length[myid]
            , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    localMat->nnz = val_local_length[myid];
#ifdef DEBUG_DATA_DIST_2
    // how to check this scatterv has been done correctly ?
    //, use small spd and check manually
    int temp_idx;
    printf("in SparseMatrixDistribution.c, myid=%d\t\t", myid);
    //    for (temp_idx=0; temp_idx<col_local_length[myid]; temp_idx++){
    //        printf( "%f ",localMat->csrdata[temp_idx]);
    //    }
    for (temp_idx = 0; temp_idx < col_local_length[myid]; temp_idx++) {
        printf("%d ", localMat->col_idx[temp_idx]);
    }
    printf("distributed data print ending\n");
#endif
    // finally gabage management

    free(rank_0_global_csr.row_start);


    free(col_local_start_idx);
    free(col_local_length);
    free(val_local_start_idx);
    free(val_local_length);

}

