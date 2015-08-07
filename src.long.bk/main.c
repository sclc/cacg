#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "DataTypes.h"
#include "common.h"
#include "RHSgen.h"
#include "bcgKernel.h"
#include "cbcgKernel.h"
#include "bcbcgKernel.h"
#include "matrixType.h"
#include "SparseMatrixDistribution.h"
#include "GetRHS.h"

//#define KSMs_DEBUG 
// #define CAL_GERSCHGORIN
#define CAL_LS

int main(int argc, char* argv[]) {

    int myid, numprocs;
    long solverIdx;
    long sVal;
    char * path;
    char * mtx_filename;
    char * rhs_filename;
    char * string_num_cols;
    char * str_solverIdx;
    char * str_sVal;

    double epsilon = 1e-15;

    double t1, t2, t_past_local, t_past_global;

    if (argc < 7) {
        printf("Argument setting is wrong.\n");
        exit(0);
    } else {
        path = argv[1];
        mtx_filename = argv[2];
        rhs_filename = argv[3];
        string_num_cols = argv[4];
        str_solverIdx = argv[5];
        str_sVal = argv[6];
    }

    long set_num_cols;
    set_num_cols = (long)atoi(string_num_cols);


    solverIdx = (long)atoi(str_solverIdx);

    sVal = (long)atoi(str_sVal);

    int ierr;


    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);   
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (myid == 0) {
        printf("we can want to generate RHS with size of %d\n", set_num_cols);
    }

// printf ("%ld, %ld, %ld", set_num_cols, solverIdx, sVal);
    // exit(0);

    csrType_local local_Mat;
    denseType B;
    denseType X;
    matInfo mat_info;


    Sparse_Csr_Matrix_Distribution(&local_Mat, &mat_info, myid, numprocs
            , path, mtx_filename);
   // printf("aabba %lf\n",local_Mat.csrdata[0]);
   

        // printf("so far so good AEE\n");
        // exit(0);

    // generate RHS B
//        GenVectorOne(mat_info.num_rows, &B, set_num_cols,myid, numprocs);
    GenVector_ReadCSV(&B, mat_info.num_rows, set_num_cols, rhs_filename, myid, numprocs);
#ifdef KSMs_DEBUG
    if (myid == numprocs - 1) {
        local_dense_mat_print(B, myid);
    }
#endif
    // generate storage for unknown matrix X
    GenVectorOne(mat_info.num_rows, &X, set_num_cols, myid, numprocs);
    // GenVectorRandom(mat_info.num_rows, &X, set_num_cols, 0.0, 10.0, myid, numprocs);


#ifdef KSMs_DEBUG
    //    printf ("in main.c, rank= %d, mat_info.num_cols=%d\n", myid,mat_info.num_cols);
#endif

#ifdef CAL_GERSCHGORIN
    double eigenValue [2];
    gerschgorin_v1(eigenValue, local_Mat);
    printf ("min: %20.16f, max:%20.16f\n", eigenValue[0], eigenValue[1]);
#endif
    
#ifdef CAL_LS
    switch (solverIdx) {
        case 0: //BCG
            if (myid == 0) {
                printf("Going to call BCG solver ... ...\n");
                printf("m value:%d\n", set_num_cols);
            }
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
//            bcg_v1(local_Mat, B, X, epsilon, myid, numprocs);
            bcg_QR(local_Mat, B, X, epsilon, myid, numprocs);
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t2 = MPI_Wtime();
            t_past_local = t2 - t1;

            // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
            //    MPI_Op op, int root, MPI_Comm comm)
            ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                printf("Total time for BCG solver: %f secs\n", t_past_global);
            }
            break;
        case 1: //CBCG
            if (myid == 0) {
                printf("Going to call CBCG solver ... ...\n");
                printf("s value:%d; m value:%d\n", sVal, set_num_cols);
            }
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            cbcg_v1(local_Mat, B, X, sVal, epsilon, myid, numprocs);
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t2 = MPI_Wtime();
            t_past_local = t2 - t1;
            ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                printf("Total time for CBCG solver: %f secs\n", t_past_global);
            }
            break;
        case 2: //BCBCG
            if (myid == 0) {
                printf("Going to call BCBCG solver ... ...\n");
                printf("s value:%d; m value:%d\n", sVal, set_num_cols);
            }
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            bcbcg_v1(local_Mat, B, X, sVal, epsilon, myid, numprocs);
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            t2 = MPI_Wtime();
            t_past_local = t2 - t1;
            ierr = MPI_Reduce(&t_past_local, &t_past_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0) {
                printf("Total time for BCBCG solver: %f secs\n", t_past_global);
            }
            break;
        default:
            printf("Sorry, wrong solver index\n");
            break;
    }
#endif
    //Garbage collection
    delete_denseType(B);
    delete_denseType(X);
    // do not forget to clean csr matrix

    ierr = MPI_Finalize();

    return 0;

}
