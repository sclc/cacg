// #define DEBUG_DATA_DIST
// #define DEBUG_DATA_DIST_2
//#define DEBUG_DATA_DIST_LOCAL_ROW_START

#include "SparseMatrixDistribution.h"

void Sparse_Csr_Matrix_Distribution(csrType_local * localMat, matInfo * mat_info \
                                  , int myid, int numprocs \
                                  , char* path, char* mtx_filename) {

    // use MPI blocking communication
    cooType globalMat;

    csrType_local rank_0_global_csr;


    int * col_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    // long * col_local_start_idx = (long *) calloc(numprocs, sizeof (long));
    // size_t * col_local_start_idx = (size_t *) calloc(numprocs, sizeof (size_t));

    int * col_local_length = (int *) calloc(numprocs, sizeof (int));
    // long * col_local_length = (long *) calloc(numprocs, sizeof (long));
    // size_t * col_local_length = (size_t *) calloc(numprocs, sizeof (size_t));

    // long * val_local_start_idx = (long *) calloc(numprocs, sizeof (long));
    // long * val_local_length = (long *) calloc(numprocs, sizeof (long));
    int * val_local_start_idx = (int *) calloc(numprocs, sizeof (int));
    int * val_local_length = (int *) calloc(numprocs, sizeof (int));

    long rows_per_proc, num_total, num_row_local;

    long idx;
    int ierr;

    long num_row_col[2];


    // printf("long myid is %ld\n", myid);
    if (myid == 0) {
        // rank 0 read mtr

        readMtx_info_and_coo(path, mtx_filename, mat_info, & globalMat);
        // printf("so far so good C\n");
        // exit(0);

        printf("there are %ld elements in matrix \n", mat_info->nnz);
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
    MPI_Bcast((void *) num_row_col, 2, MPI_LONG, 0, MPI_COMM_WORLD);
    num_total = num_row_col[0];
    rows_per_proc = num_total / numprocs;


    if (myid != 0) {
        rank_0_global_csr.row_start = (long *) calloc(num_total + 1, sizeof (long));
    }
#ifdef DEBUG_DATA_DIST
    printf("myid=%d, num_row_col[0]=%ld, num_row_col[1]=%ld \n"
            , myid, num_row_col[0], num_row_col[1]);
#endif


    if (myid == (numprocs - 1)) {
        num_row_local = num_total - rows_per_proc * (numprocs - 1);
    } else {
        num_row_local = rows_per_proc;
    }

    localMat->num_rows = num_row_local;

    localMat->num_cols = num_row_col[1];

    localMat->row_start = (long *) calloc(num_row_local + 1, sizeof (long));
    localMat->start = myid * rows_per_proc;
#ifdef DEBUG_DATA_DIST
    //    printf("myid=%d, localMat->start=%d, localMat->num_rows=%d\n", myid, localMat->start, localMat->num_rows);
#endif    

    ierr = MPI_Bcast((void*) rank_0_global_csr.row_start, num_total + 1, MPI_LONG
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
        long row_start = idx * rows_per_proc;
        long row_end = row_start + rows_per_proc;
        if (idx == (numprocs - 1))
            row_end = num_total;

        //col_local_start_idx[idx] is int, check rank_0_global_csr.row_start[row_start] value range
        // col_local_start_idx[idx] = (int)rank_0_global_csr.row_start[row_start];
        col_local_start_idx[idx] = (int)rank_0_global_csr.row_start[row_start];
        // col_local_start_idx[idx] = (size_t)rank_0_global_csr.row_start[row_start];

        col_local_length[idx] = (int)rank_0_global_csr.row_start[row_end]
                - rank_0_global_csr.row_start[row_start];

        val_local_start_idx[idx] = (int)rank_0_global_csr.row_start[row_start];
        val_local_length[idx] = (int)(rank_0_global_csr.row_start[row_end]
                - rank_0_global_csr.row_start[row_start]);
    }
 
#ifdef DEBUG_DATA_DIST
    long total_entries = 0;
    for (idx = 0; idx < numprocs; idx++)
        total_entries += col_local_length[idx];
    printf("#########myid=%d, total_entries=%ld\n", myid, total_entries);
    printf("#########myid=%d, local_entries=%ld\n", myid, col_local_length[myid]);
    printf("#########myid=%d, val_local_length=%ld\n", myid, val_local_length[myid]);

#endif


    localMat->col_idx = (long *) calloc(col_local_length[myid], sizeof (long));
    localMat->csrdata = (double*) calloc(val_local_length[myid], sizeof (double));

    // printf("before MPI_Scatterv\n");
    // printf ("rank_0_global_csr.col_idx[3]: %ld\n", rank_0_global_csr.col_idx[3]);

    // col index scatterv
    ierr = MPI_Scatterv((void *) rank_0_global_csr.col_idx, (int*)col_local_length
            , (int*)col_local_start_idx, MPI_LONG
            , (void *) localMat->col_idx, (int)col_local_length[myid]
            , MPI_LONG, 0, MPI_COMM_WORLD);

    // printf("after MPI_Scatterv\n");
    // printf ("localMat->col_idx[3]: %ld\n", localMat->col_idx[3]);
    // printf("size of size_t, int, long, double: %d, %d, %d, %d\n"\
    //      , sizeof(size_t), sizeof(int), sizeof(long), sizeof(double));
    // exit(0);

    // val scatterv
    // int MPI_Scatterv(const void *sendbuf, const int *sendcounts, const int *displs,
                 // MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 // MPI_Datatype recvtype,
                 // int root, MPI_Comm comm)
    // runtime data wrongly distributed, if use long type displs
    ierr = MPI_Scatterv((void *) rank_0_global_csr.csrdata, (int*)val_local_length
            , (int*)val_local_start_idx, MPI_DOUBLE
            , (void *) localMat->csrdata, (int)val_local_length[myid]
            , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    localMat->nnz = val_local_length[myid];
    // printf ("localMat->nnz : %ld\n", val_local_length[myid]);

#ifdef DEBUG_DATA_DIST_2
    // how to check this scatterv has been done correctly ?
    //, use small spd and check manually
    long temp_idx;
    printf("in SparseMatrixDistribution.c, myid=%d\t\t", myid);
    //    for (temp_idx=0; temp_idx<col_local_length[myid]; temp_idx++){
    //        printf( "%f ",localMat->csrdata[temp_idx]);
    // }
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

