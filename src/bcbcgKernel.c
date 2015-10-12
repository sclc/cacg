#include "bcbcgKernel.h"

#define MAXITER 100
#define NUM_LOOP_PER_PRINT 10

// #define bcbcg_v1_DB_Ger
 // #define bcbcg_v1_DB_R
//#define bcbcg_v1_DB_CPBG
//#define bcbcg_v1_DB_AQ
//#define bcbcg_v1_DB_QtAQ
//#define bcbcg_v1_DB_ALPHA
//#define bcbcg_v1_DB_AQALPHA
//#define bcbcg_v1_DB_UpdatedX
//#define bcbcg_v1_DB_UpdatedR
// #define bcbcg_v1_DB_LONG
// #define LU_MAT_RAND
// #define DB_SIG_FAULT

#define TIME_BCBCG_PROFILING
#define TIME_MEASURE_BCBCG_CB_GEN
#define TIME_MEASURE_BCBCG_IP_COLLECTIVE
#define TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
#define TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
#define TIME_MEASURE_BCBCG_MAT_INV_LOCAL
// s: s-step

void bcbcg_v1(csrType_local mat, denseType B, denseType X, \
              long s, double epsilon, int myid, int numprocs) {

#ifdef TIME_BCBCG_PROFILING
    int ierr;
#endif

#ifdef TIME_MEASURE_BCBCG_CB_GEN
        double cb_gen_timer_1, cb_gen_timer_2;
        double cb_gen_local=0.0, cb_gen_global=0.0;
#endif
#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
        double ip_collective_timer_1, ip_collective_timer_2;
        double ip_collective_local=0.0, ip_collective_global=0.0;
#endif
#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
        double spmm_collective_timer_1, spmm_collective_timer_2;
        double spmm_collective_local=0.0, spmm_collective_global=0.0;
#endif
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        double mat_update_local_timer1, mat_update_local_timer2;
        double mat_update_local=0.0, mat_update_global=0.0;
#endif
#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        double mat_inv_local_timer1, mat_inv_local_timer2;
        double mat_inv_local=0.0,mat_inv_global=0.0;        
#endif

    denseType R;
    get_same_shape_denseType(B, &R);

    denseType A_X;
    get_same_shape_denseType(B, &A_X);

    double * B_global_shape_swap_zone = (double*) calloc(B.global_num_col * B.global_num_row, sizeof (double));

    // S buffer will also be distributed
    denseType S_Mat;
    gen_dense_mat(&S_Mat, X.local_num_row, s * B.local_num_col
            , X.global_num_row, s * B.global_num_col
            , mat.start);

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

//debug point 
#ifdef bcbcg_v1_DB_Ger
    printf("max:%f,min:%f\n", eigenValBounds[1], eigenValBounds[0]);
#endif

    double s_alpha = 2.0 / (eigenValBounds[1] - eigenValBounds[0]);
    double s_beta = -(eigenValBounds[1] + eigenValBounds[0])
            / (eigenValBounds[1] - eigenValBounds[0]);

//debug point 
#ifdef bcbcg_v1_DB_Ger
    printf("s_alpha:%f,s_beta:%f\n", s_alpha, s_beta);
    exit(1);
#endif

    //A*X, Spmm when m=1
#ifdef bcbcg_v1_DB_LONG
    local_dense_mat_print(X, myid);
    // exit(0);
#endif

#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
    // for the time being do not contain this part
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
    // spmm_collective_timer_1 = MPI_Wtime();
#endif


    spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);


#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
    // for the time being do not contain this part
    // ierr = MPI_Barrier(MPI_COMM_WORLD);
    // spmm_collective_timer_2 = MPI_Wtime();
    // spmm_collective_local += spmm_collective_timer_2 - spmm_collective_timer_1;
#endif

#ifdef bcbcg_v1_DB_LONG
    double db_res_ax;
    norm2square_dist_denseMat_col_n(A_X, &db_res_ax, 1, myid, numprocs);
    printf ("AX(1) norm is %lf\n",db_res_ax);
    // exit(0);
#endif
    //R=B-AX
    // to comment
    dense_mat_mat_add_TP(B, A_X, R, 1.0, -1.0, myid, numprocs);
#ifdef bcbcg_v1_DB_R
    // if (myid == numprocs - 1) {
    if (myid == 0) {
        // local_dense_mat_print(R, myid);
        double db_res;
        // norm2square_dist_denseMat_col_n(R, &db_res, 1, myid, numprocs);
        // printf ("R0(1) norm is %lf\n",db_res);
        // exit(0);
    }
    exit(1);
#endif

    // start looping phrase for BCBCG
    // 
    //
    long maxIters = MAXITER;
    long iterCounter;
    for (iterCounter = 0; iterCounter < maxIters; iterCounter++) {

        //To generate S Chebyshev basis, namely, update S with new R
#ifdef TIME_MEASURE_BCBCG_CB_GEN
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        cb_gen_timer_1 = MPI_Wtime();
#endif

        BCBCG_chebyshevPolynomialBasisGen(S_Mat, mat, R, s, s_alpha, s_beta, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_CB_GEN
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        cb_gen_timer_2 = MPI_Wtime();
        cb_gen_local += cb_gen_timer_2 - cb_gen_timer_1;
#endif

#ifdef bcbcg_v1_DB_CPBG
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            local_dense_mat_print(S_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // useless ?
        // MPI_Barrier(MPI_COMM_WORLD);
        //update Q
        if (iterCounter == 0) {
            free(Q_Mat.data);
            Q_Mat.data = S_Mat.data;
            S_Mat.data = (double *) calloc(S_Mat.local_num_col * S_Mat.local_num_row, sizeof (double));
        } 
        else 
        {
            // calculate QtAS by using old AQ and new S
            // A is SPD, so QtA = (AQ)**t
            // QtAS = (AQ)**t * S, sort of inner product
#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_1 = MPI_Wtime();
#endif
            // to comment
            distributedMatLTransposeMatRMul_updated(&QtAS_Mat, AQ_Mat, S_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_2 = MPI_Wtime();
            ip_collective_local += ip_collective_timer_2 - ip_collective_timer_1;

#endif
            // using old QtAQ and new QtAS
            // solve QtAQ * beta = QtAS to get beta
#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_inv_local_timer1 = MPI_Wtime();     
#endif
            // to comment
            LUsolver_v1_KeeplhsMat(&beta_Mat, QtAQ_Mat, QtAS_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_inv_local_timer2 = MPI_Wtime();
        mat_inv_local += mat_inv_local_timer2 - mat_inv_local_timer1;     
#endif
#ifdef LU_MAT_RAND
        long firstInvRow = beta_Mat.local_num_row;
        long firstInvCol = beta_Mat.local_num_col;
        free(beta_Mat.data);

        Local_Dense_Mat_Generator( &beta_Mat, firstInvRow, firstInvCol,0.0,10.0);
#endif
            // Q * beta: using old Q and new beta
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
            // to comment
            dense_distributedMatL_localMatR_Mul_v1(&Qbeta_Mat, Q_Mat, beta_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif
            // update Q: Q = S - Qbeta
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
            // to comment
            dense_mat_mat_add_TP(S_Mat, Qbeta_Mat, Q_Mat, 1.0, -1.0, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif

        }

        // A*Q
#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        spmm_collective_timer_1 = MPI_Wtime();
#endif

#ifdef DB_SIG_FAULT
        if (myid == 0 && iterCounter == 80) {
            local_dense_mat_print(Q_Mat, myid);
            //            local_dense_mat_print(Q_Mat, myid);
        }
#endif
        spmm_csr_v2(mat, Q_Mat, &AQ_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
        ierr = MPI_Barrier(MPI_COMM_WORLD);
        spmm_collective_timer_2 = MPI_Wtime();
        spmm_collective_local += spmm_collective_timer_2 - spmm_collective_timer_1;
#endif

#ifdef bcbcg_v1_DB_AQ
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            local_dense_mat_print(AQ_Mat, myid);
            //            local_dense_mat_print(Q_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        //QtAQ
#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_1 = MPI_Wtime();
#endif
        // to comment
        distributedMatLTransposeMatRMul(&QtAQ_Mat, Q_Mat, AQ_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_2 = MPI_Wtime();
            ip_collective_local += ip_collective_timer_2 - ip_collective_timer_1;
#endif

#ifdef bcbcg_v1_DB_QtAQ
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            //            printf("%d,%d\n",QtAQ_Mat.local_num_col, QtAQ_Mat.local_num_row);
            local_dense_mat_print(QtAQ_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        //QtR_Mat
#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_1 = MPI_Wtime();
#endif
        // to comment
        distributedMatLTransposeMatRMul_updated(&QtR_Mat, Q_Mat, R, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            ip_collective_timer_2 = MPI_Wtime();
            ip_collective_local += ip_collective_timer_2 - ip_collective_timer_1;
#endif

        //solve QtAQ * alpha = QtR to get alpha
#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_inv_local_timer1 = MPI_Wtime();     
#endif
        // to comment
        LUsolver_v1_KeeplhsMat(&alpha_Mat, QtAQ_Mat, QtR_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_inv_local_timer2 = MPI_Wtime();
        mat_inv_local += mat_inv_local_timer2 - mat_inv_local_timer1;     
#endif

//debug point    
#ifdef LU_MAT_RAND
        long secondInvRow = alpha_Mat.local_num_row;
        long secondInvCol = alpha_Mat.local_num_col;
        free(alpha_Mat.data);

        Local_Dense_Mat_Generator( &alpha_Mat, secondInvRow, secondInvCol,0.0,10.0);
#endif

//debug point       
#ifdef bcbcg_v1_DB_ALPHA
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            local_dense_mat_print(alpha_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // Q * alpha
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
        // to comment
        dense_distributedMatL_localMatR_Mul_v1(&Qalpha_Mat, Q_Mat, alpha_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif

        // AQ * alpha
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
        // to comment
        dense_distributedMatL_localMatR_Mul_v1(&AQalpha_Mat, AQ_Mat, alpha_Mat, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif

//debug point 
#ifdef bcbcg_v1_DB_AQALPHA
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            local_dense_mat_print(AQalpha_Mat, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif

        // X = X+Qalpha_Mat
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
        // to comment
        dense_mat_mat_add_TP(X, Qalpha_Mat, X, 1.0, 1.0, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif

//debug point 
#ifdef bcbcg_v1_DB_UpdatedX
    // if (myid == numprocs - 1) {
    if (myid == 0) {
            local_dense_mat_print(X, myid);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(1);
#endif
        // R = R - AQalpha
#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer1 = MPI_Wtime();
#endif
        // to comment
        dense_mat_mat_add_TP(R, AQalpha_Mat, R, 1.0, -1.0, myid, numprocs);

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        // ierr = MPI_Barrier(MPI_COMM_WORLD);
        mat_update_local_timer2 = MPI_Wtime();
        mat_update_local += mat_update_local_timer2 - mat_update_local_timer1;
#endif
        // check vector norm
        long n_col = 1;
        double R_norm2squre;
        norm2square_dist_denseMat_col_n(R, &R_norm2squre, n_col, myid, numprocs);

        if (myid == 0 &&  (iterCounter % NUM_LOOP_PER_PRINT==0 || sqrt(R_norm2squre) < epsilon \
            || iterCounter==(maxIters-1))) {
            printf("Loop: %d, R[%d]_norm=%30.30f\n", iterCounter, n_col, sqrt(R_norm2squre));
        }

#ifdef TIME_MEASURE_BCBCG_CB_GEN
        if(sqrt(R_norm2squre) < epsilon || iterCounter==(maxIters-1) )
        {
            cb_gen_local = cb_gen_local / iterCounter;
            ierr = MPI_Reduce(&cb_gen_local, &cb_gen_global, 1, \
                              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0)
            {
                printf("Loop: %d, time_BCBCG_chebyshevPolynomialBasisGen=%30.30f\n", \
                        iterCounter, cb_gen_global);
            }
        }
#endif

#ifdef TIME_MEASURE_BCBCG_IP_COLLECTIVE
        if(sqrt(R_norm2squre) < epsilon || iterCounter==(maxIters-1) )
        {
            ip_collective_local = ip_collective_local / iterCounter;
            ierr = MPI_Reduce(&ip_collective_local, &ip_collective_global, 1, \
                              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0)
            {
                printf("Loop: %d, time_BCBCG_innerProduct_collective=%30.30f\n", \
                        iterCounter, ip_collective_global);
            }
        }
#endif

#ifdef TIME_MEASURE_BCBCG_SPMM_COLLECTIVE
        if(sqrt(R_norm2squre) < epsilon || iterCounter==(maxIters-1) )
        {
            spmm_collective_local = spmm_collective_local / iterCounter;
            ierr = MPI_Reduce(&spmm_collective_local, &spmm_collective_global, 1, \
                              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0)
            {
                printf("Loop: %d, time_BCBCG_spmm_collective=%30.30f\n", \
                        iterCounter, spmm_collective_global);
            }
        }
#endif

#ifdef TIME_MEASURE_BCBCG_MAT_UPDATE_LOCAL
        if(sqrt(R_norm2squre) < epsilon || iterCounter==(maxIters-1) )
        {
            mat_update_local = mat_update_local / iterCounter;
            ierr = MPI_Reduce(&mat_update_local, &mat_update_global, 1, \
                              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0)
            {
                printf("Loop: %d, time_BCBCG_mat_update_local=%30.30f\n", \
                        iterCounter, mat_update_global);
            }
        }
#endif

#ifdef TIME_MEASURE_BCBCG_MAT_INV_LOCAL
        if(sqrt(R_norm2squre) < epsilon || iterCounter==(maxIters-1) )
        {
            mat_inv_local = mat_inv_local / iterCounter;
            ierr = MPI_Reduce(&mat_inv_local, &mat_inv_global, 1, \
                              MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (myid == 0)
            {
                printf("Loop: %d, time_BCBCG_mat_inv_local=%30.30f\n", \
                        iterCounter, mat_inv_global);
            }
        }      
#endif

        if (sqrt(R_norm2squre) < epsilon) {
            return;
        }
//debug point 
#ifdef bcbcg_v1_DB_UpdatedR
    // if (myid == numprocs - 1) {
    if (myid == 0) {
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

