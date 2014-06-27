#include "cbcgKernel.h"

//#define cbcg_v1_DB_Ger
//#define cbcg_v1_DB_R
//#define cbcg_v1_DB_CPBG
//#define cbcg_v1_DB_AQ
//#define cbcg_v1_DB_QtAQ
//#define cbcg_v1_DB_ALPHA
//#define cbcg_v1_DB_AQALPHA
//#define cbcg_v1_DB_UpdatedX
//#define cbcg_v1_DB_UpdatedR
// s: s-step

void cbcg_v1(csrType_local mat, denseType B, denseType X, int s, double epsilon, int myid, int numprocs) {

    denseType R;
    get_same_shape_denseType(B, &R);

    denseType A_X;
    get_same_shape_denseType(B, &A_X);

    double * B_global_shape_swap_zone = (double*) calloc(B.global_num_col * B.global_num_row, sizeof (double));

    // S buffer will also be distributed
    denseType S_Mat;
    gen_dense_mat(&S_Mat, X.local_num_row, s, X.global_num_row, s, mat.start);

    denseType Q_Mat;
    get_same_shape_denseType(S_Mat, &Q_Mat);

    denseType AQ_Mat;
    get_same_shape_denseType(S_Mat, &AQ_Mat);

    denseType QtAQ_Mat;
    generate_no_parallel_dense(&QtAQ_Mat, Q_Mat.local_num_col, Q_Mat.local_num_col);

    denseType QtAS_Mat; // QtAS and QtAQ have the same shape
    get_same_shape_denseType(QtAQ_Mat, &QtAS_Mat);

    denseType QtR_Mat;
    generate_no_parallel_dense(&QtR_Mat, Q_Mat.local_num_col, R.global_num_col);

    denseType alpha_Mat;
    generate_no_parallel_dense(&alpha_Mat, QtAQ_Mat.local_num_col, QtR_Mat.local_num_col);

    denseType beta_Mat;
    generate_no_parallel_dense(&beta_Mat, Q_Mat.global_num_col, S_Mat.global_num_col);

    denseType Qalpha_Mat;
    get_same_shape_denseType(X, &Qalpha_Mat);

    denseType AQalpha_Mat;
    get_same_shape_denseType(X, &AQalpha_Mat);

    //Q_Mat.start_idx may be undefined ?
    denseType Qbeta_Mat;
    gen_dense_mat(&Qbeta_Mat, Q_Mat.local_num_row, beta_Mat.local_num_col
            , Q_Mat.global_num_row, beta_Mat.global_num_col, Q_Mat.start_idx);

    // shift matrix item values using max and min eigen-value
    // gerschgorin function call
    // output[0]=min; output[1]=max;
    double eigenValBounds[2];
    gerschgorin_v1(eigenValBounds, mat);
#ifdef cbcg_v1_DB_Ger
    printf("max:%f,min:%f\n", eigenValBounds[1], eigenValBounds[0]);
#endif

    double s_alpha = 2.0 / (eigenValBounds[1] - eigenValBounds[0]);
    double s_beta = -(eigenValBounds[1] + eigenValBounds[0])
            / (eigenValBounds[1] - eigenValBounds[0]);
#ifdef cbcg_v1_DB_Ger
    printf("s_alpha:%f,s_beta:%f\n", s_alpha, s_beta);
    exit(1);
#endif
    //A*X, Spmm when m=1
    spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);
    //R=B-AX
    dense_mat_mat_add_TP(B, A_X, R, 1.0, -1.0, myid, numprocs);
#ifdef cbcg_v1_DB_R
    if (myid == numprocs - 1) {
        local_dense_mat_print(R, myid);
    }
    exit(1);
#endif
    int maxIters = 1000;
    int iterCounter;
    for (iterCounter = 0; iterCounter < maxIters; iterCounter++) {
        //To generate S Chebyshev basis, namely, update S with new R
        CBCG_chebyshevPolynomialBasisGen(S_Mat, mat, R, s, s_alpha, s_beta, myid, numprocs);
#ifdef cbcg_v1_DB_CPBG
        if (myid == numprocs - 1) {
            local_dense_mat_print(S_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // useless ?
        MPI_Barrier(MPI_COMM_WORLD);
        //update Q
        if (iterCounter == 0) {
            free(Q_Mat.data);
            Q_Mat.data = S_Mat.data;
            S_Mat.data = (double *) calloc(S_Mat.local_num_col * S_Mat.local_num_row, sizeof (double));
        } else {
            // calculate QtAS by using old AQ and new S
            // A is SPD, so QtA = (AQ)**t
            // QtAS = (AQ)**t * S, sort of inner product
            distributedMatLTransposeMatRMul_updated(&QtAS_Mat, AQ_Mat, S_Mat, myid, numprocs);

            // using old QtAQ and new QtAS
            // solve QtAQ * beta = QtAS to get beta
            LUsolver_v1_KeeplhsMat(&beta_Mat, QtAQ_Mat, QtAS_Mat, myid, numprocs);

            // Q * beta: using old Q and new beta
            dense_distributedMatL_localMatR_Mul_v1(&Qbeta_Mat, Q_Mat, beta_Mat, myid, numprocs);

            // update Q: Q = S - Qbeta
            dense_mat_mat_add_TP(S_Mat, Qbeta_Mat, Q_Mat, 1.0, -1.0, myid, numprocs);

        }
        // A*Q
        spmm_csr_v2(mat, Q_Mat, &AQ_Mat, myid, numprocs);
#ifdef cbcg_v1_DB_AQ
        if (myid == numprocs - 1) {
            local_dense_mat_print(AQ_Mat, myid);
            //            local_dense_mat_print(Q_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        //QtAQ
        distributedMatLTransposeMatRMul(&QtAQ_Mat, Q_Mat, AQ_Mat, myid, numprocs);
#ifdef cbcg_v1_DB_QtAQ
        if (myid == 0) {
            //            printf("%d,%d\n",QtAQ_Mat.local_num_col, QtAQ_Mat.local_num_row);
            local_dense_mat_print(QtAQ_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        //QtR_Mat
        distributedMatLTransposeMatRMul_updated(&QtR_Mat, Q_Mat, R, myid, numprocs);
        //solve QtAQ * alpha = QtR to get alpha
        LUsolver_v1_KeeplhsMat(&alpha_Mat, QtAQ_Mat, QtR_Mat, myid, numprocs);
#ifdef cbcg_v1_DB_ALPHA
        if (myid == 0) {
            local_dense_mat_print(alpha_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // Q * alpha
        dense_distributedMatL_localMatR_Mul_v1(&Qalpha_Mat, Q_Mat, alpha_Mat, myid, numprocs);
        // AQ * alpha
        dense_distributedMatL_localMatR_Mul_v1(&AQalpha_Mat, AQ_Mat, alpha_Mat, myid, numprocs);
#ifdef cbcg_v1_DB_AQALPHA
        if (myid == 2) {
            local_dense_mat_print(AQalpha_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif

        // X = X+Qalpha_Mat
        dense_mat_mat_add_TP(X, Qalpha_Mat, X, 1.0, 1.0, myid, numprocs);
#ifdef cbcg_v1_DB_UpdatedX
        if (myid == 2) {
            local_dense_mat_print(X, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // R = R - AQalpha
        dense_mat_mat_add_TP(R, AQalpha_Mat, R, 1.0, -1.0, myid, numprocs);
        // check vector norm
        int n_col = 1;
        double R_norm2squre;
        norm2square_dist_denseMat_col_n(R, &R_norm2squre, n_col, myid, numprocs);
        if (myid == 0) {
            printf("Loop: %d, R[%d]_norm=%30.30f\n", iterCounter, n_col, sqrt(R_norm2squre));
        }
        if (sqrt(R_norm2squre) < epsilon) {
            return;
        }
#ifdef cbcg_v1_DB_UpdatedR
        if (myid == 2) {
            local_dense_mat_print(R, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
    }


    //memory garbage collection 
    delete_denseType(R);
    delete_denseType(A_X);
    delete_denseType(S_Mat);
    delete_denseType(Q_Mat);
    delete_denseType(AQ_Mat);
    delete_denseType(QtAQ_Mat);
    delete_denseType(QtR_Mat);
    delete_denseType(alpha_Mat);
    delete_denseType(beta_Mat);
    delete_denseType(Qalpha_Mat);
    delete_denseType(Qbeta_Mat);
    delete_denseType(AQalpha_Mat);
    delete_denseType(QtAS_Mat);


    free(B_global_shape_swap_zone);

}