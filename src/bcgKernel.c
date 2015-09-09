#include "bcgKernel.h"
#define BCG_MAXITER 3200
#define NUM_LOOP_PER_PRINT 10

//#define BCG_V1_DEBUG
//#define BCG_V1_DEBUG_1
//#define BCG_V1_DEBUG_2
//#define BCG_V1_DEBUG_NORM_CHECK
//#define BCG_V1_DEBUG_3
//#define BCG_V1_DEBUG_PtAP
//#define BCG_V1_DEBUG_RtR
//#define BCG_V1_DEBUG_LU
//#define BCG_V1_DEBUG_P_ALPHA
//#define BCG_V1_DEBUG_P_NEW
//#define BCG_V1_DEBUG_AP

void bcg_v1(csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs) {
    long ierr;
    denseType R;
    get_same_shape_denseType(B, &R);

    long mat_global_row_num = mat.num_cols;
    long mat_global_col_num = mat.num_cols;

    double * B_global_shape_swap_zone = (double*) calloc(B.global_num_col * B.global_num_row, sizeof (double));

    denseType A_X;
    get_same_shape_denseType(B, &A_X);
    //    set_dense_to_zero(A_X);

    denseType P;
    get_same_shape_denseType(B, &P);

    denseType A_P;
    get_same_shape_denseType(B, &A_P);

    denseType P_alpha;
    get_same_shape_denseType(B, &P_alpha);

    denseType A_P_alpha;
    get_same_shape_denseType(B, &A_P_alpha);

    denseType P_beta;
    get_same_shape_denseType(B, &P_beta);

#ifdef BCG_V1_DEBUG
    long idx_debug;
    printf("in bcg_V1, myid=%d, ", myid);
    for (idx_debug = 0; idx_debug < A_X.local_num_col * A_X.local_num_row; idx_debug++) {
        if (A_X.data[idx_debug] != 0.0)
            printf("%f wrong \n", A_X.data[idx_debug]);
    }

    long X_num_data = X.local_num_col * X.local_num_row;
    printf("in bcgKernel.c, myid=%d, X_num_data=%d\n", myid, X_num_data);
    for (idx_debug = 0; idx_debug < X_num_data; idx_debug++) {
        if (X.data[idx_debug] != 1.0) {
            printf("in bcgKernel.c, X data wrong\n");
        }
    }
#endif
    // A * X0
    spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);

#ifdef BCG_V1_DEBUG_NORM_CHECK
    double * local_norm = local_dense_colum_2norm(A_X);
    double * global_norm = (double *) calloc(A_X.global_num_col, sizeof (double));
    long temp_idx;
    printf("in bcg_v1, myid=%d ", myid);
    for (temp_idx = 0; temp_idx < A_X.local_num_col; temp_idx++) {
        printf("%f, ", local_norm[temp_idx]);
    }
    printf("local norm ending \n");

// int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
//                MPI_Op op, int root, MPI_Comm comm)
    ierr = MPI_Reduce((void*) local_norm, (void*) global_norm, (int)A_X.global_num_col
            , MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {

        printf("global squared norm array: \n");
        for (temp_idx = 0; temp_idx < A_X.global_num_col; temp_idx++) {
            printf("%f \n", global_norm[temp_idx]);
        }
        printf("norm ending\n");
    }
    free(global_norm);
    free(local_norm);
#endif
    // R0=B-AX0
#ifdef BCG_V1_DEBUG_1
    long ind_array[2] = {0, B.local_num_col * B.local_num_row - 1};
    printf("in bcg_V1, before dense_mat_mat_add_TP myid=%d ", myid);
    printf("B value: %f, %f; A_X values: %f, %f; R values: %f,%f \n"
            , B.data[ind_array[0]], B.data[ind_array[1]]
            , A_X.data[ind_array[0]], A_X.data[ind_array[1]]
            , R.data[ind_array[0]], R.data[ind_array[1]]
            );
#endif            
    dense_mat_mat_add_TP(B, A_X, R, 1.0, -1.0, myid, numprocs);
#ifdef BCG_V1_DEBUG_1
    // ind_array defined before
    printf("in bcg_V1, after dense_mat_mat_add_TP, myid=%d ", myid);
    printf("B value: %f, %f; A_X values: %f, %f; R values: %f,%f \n"
            , B.data[ind_array[0]], B.data[ind_array[1]]
            , A_X.data[ind_array[0]], A_X.data[ind_array[1]]
            , R.data[ind_array[0]], R.data[ind_array[1]]
            );
#endif
    //P0 = R0
    dense_entry_copy(R, P);
#ifdef BCG_V1_DEBUG_2
    long bcg_v1_debug_2_idx_i;
    long bcg_v1_debug_2_idx_j;
    for (bcg_v1_debug_2_idx_i = 0; bcg_v1_debug_2_idx_i < R.local_num_row; bcg_v1_debug_2_idx_i++)
        for (bcg_v1_debug_2_idx_j = 0; bcg_v1_debug_2_idx_j < R.local_num_col; bcg_v1_debug_2_idx_j++) {
            long RP_idx = bcg_v1_debug_2_idx_i * R.local_num_col + bcg_v1_debug_2_idx_j;
            assert(R.data[RP_idx] == P.data[RP_idx]);
        }
#endif

    denseType Pt_A_P;
    generate_no_parallel_dense(&Pt_A_P, P.local_num_col, P.local_num_col);
    //    denseType Pt_A_P_inv;
    //    generate_no_parallel_dense(&Pt_A_P_inv, P.local_num_col, P.local_num_col);

    denseType Rt_R;
    generate_no_parallel_dense(&Rt_R, R.local_num_col, R.local_num_col);
    denseType Rnewt_Rnew;
    generate_no_parallel_dense(&Rnewt_Rnew, R.local_num_col, R.local_num_col);

#ifdef BCG_V1_DEBUG_3
    printf("in bcg_v1, myid=%d, Pt_A_P.local_num_col=%d, Pt_A_P.local_num_row=%d \n"
            , myid, Pt_A_P.local_num_col, Pt_A_P.local_num_row);
    printf("in bcg_v1, myid=%d, Pt_A_P.data_start =%f, Pt_A_P.data_end=%f \n"
            , myid, Pt_A_P.data[0], Pt_A_P.data[Pt_A_P.local_num_col * Pt_A_P.local_num_row - 1]);
    printf("in bcg_v1, myid=%d, Rt_R.local_num_col=%d, Rt_R.local_num_row=%d \n"
            , myid, Rt_R.local_num_col, Rt_R.local_num_row);
    printf("in bcg_v1, myid=%d, Rt_R.data_start =%f, Rt_R.data_end=%f \n"
            , myid, Rt_R.data[0], Rt_R.data[Rt_R.local_num_col * Rt_R.local_num_row - 1]);

#endif
    denseType ALPHA_mat;
    generate_no_parallel_dense(&ALPHA_mat, P.local_num_col, R.local_num_col);

    denseType BETA_mat;
    generate_no_parallel_dense(&BETA_mat, R.local_num_col, R.local_num_col);


    // R0**t * R0
    MatTranposeMatMul(&Rt_R, R, myid, numprocs);

    long max_iter = BCG_MAXITER;
    long bcg_v1_loop_idx;
    for (bcg_v1_loop_idx = 0; bcg_v1_loop_idx < max_iter; bcg_v1_loop_idx++) {
        // A_P = A*P
        spmm_csr_v1(mat, P, &A_P, B_global_shape_swap_zone, myid, numprocs);
#ifdef BCG_V1_DEBUG_AP
        if (myid == 0 && bcg_v1_loop_idx == 1) {
            local_dense_mat_print(A_P, myid);
        }
#endif
        // to calculate PtAP, B_global_shape_swap_zone need to be reused.
        // so the later call should be called immediately after A_P calculation
        // PtAP
        // distributedMatTransposeLocalMatMul_v1(&Pt_A_P, A_P, B_global_shape_swap_zone, myid, numprocs);        
        distributedMatLTransposeMatRMul(&Pt_A_P, P, A_P, myid, numprocs);
#ifdef BCG_V1_DEBUG_PtAP
        if (myid == 0 && bcg_v1_loop_idx == 1) {
            local_dense_mat_print(Pt_A_P, myid);
        }
#endif
#ifdef BCG_V1_DEBUG_RtR
        if (myid == 0 && bcg_v1_loop_idx == 1) {
            local_dense_mat_print(Rt_R, myid);
        }
#endif
        // to solve linear system PtAp * alpha = RtR to get alpha
        // Pt_A_P items are changed after this call
        LUsolver_v1(&ALPHA_mat, Pt_A_P, Rt_R, myid, numprocs);
#ifdef BCG_V1_DEBUG_LU
        if (myid == 0 && bcg_v1_loop_idx == 1) {
            local_dense_mat_print(ALPHA_mat, myid);
        }
#endif
        // P * alpha
        dense_distributedMatL_localMatR_Mul_v1(&P_alpha, P, ALPHA_mat, myid, numprocs);
#ifdef BCG_V1_DEBUG_P_ALPHA
        if (myid == numprocs - 2 && bcg_v1_loop_idx == 0) {
            local_dense_mat_print(P_alpha, myid);
        }
#endif
        // X_new=X_old + P_alpha
        // wasted flops, bad function def
        dense_mat_mat_add_TP(X, P_alpha, X, 1.0, 1.0, myid, numprocs);

        // AP * alpha
        dense_distributedMatL_localMatR_Mul_v1(&A_P_alpha, A_P, ALPHA_mat, myid, numprocs);

        // check vector norm
        long n_col = 1;
        double R_norm2squre;
        norm2square_dist_denseMat_col_n(R, &R_norm2squre, n_col, myid, numprocs);
        if (myid == 0 && (bcg_v1_loop_idx % NUM_LOOP_PER_PRINT==0 || sqrt(R_norm2squre) < epsilon \
            || bcg_v1_loop_idx == (max_iter-1))) {
            printf("Loop: %d, R[%d]_norm=%30.30f\n", bcg_v1_loop_idx, n_col, sqrt(R_norm2squre));
        }
        if (sqrt(R_norm2squre) < epsilon) {
            return;
        }
        // R_new = R_old - AP_alpha
        dense_mat_mat_add_TP(R, A_P_alpha, R, 1.0, -1.0, myid, numprocs);

        // calculate R_new **t * R_new
        MatTranposeMatMul(&Rnewt_Rnew, R, myid, numprocs);
        // solve R_old**t * R_old * beta = R_new**t * R_new to for beta
        // Rt_R item values are changed after this LUsolver_v1 call
        LUsolver_v1(&BETA_mat, Rt_R, Rnewt_Rnew, myid, numprocs);

        if (Rt_R.data != 0) {
            free(Rt_R.data);
        } else {
            exit(0);
        }

        // exchange data zone
        Rt_R.data = (double*) Rnewt_Rnew.data;
        Rnewt_Rnew.data = (void*) 0;

        // P_beta= P * beta
        dense_distributedMatL_localMatR_Mul_v1(&P_beta, P, BETA_mat, myid, numprocs);
        // P_new = R_new + P_beta
        // wasted flops, bad function def
        dense_mat_mat_add_TP(R, P_beta, P, 1.0, 1.0, myid, numprocs);
#ifdef BCG_V1_DEBUG_P_NEW
        if (myid == 0 && bcg_v1_loop_idx == 5) {
            printf("In loop %d, P_new elements are: \n", bcg_v1_loop_idx);
            local_dense_mat_print(P, myid);
        }
#endif

    }

    // garbage collection
    free(B_global_shape_swap_zone);
}

void bcg_QR(csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs) {
    long ierr;
    denseType R;
    get_same_shape_denseType(B, &R);

    long mat_global_row_num = mat.num_cols;
    long mat_global_col_num = mat.num_cols;

    double * B_global_shape_swap_zone = (double*) calloc(B.global_num_col * B.global_num_row, sizeof (double));

    denseType A_X;
    get_same_shape_denseType(B, &A_X);
    //    set_dense_to_zero(A_X);

    denseType P;
    get_same_shape_denseType(B, &P);

    denseType A_P;
    get_same_shape_denseType(B, &A_P);

    denseType P_alpha;
    get_same_shape_denseType(B, &P_alpha);

    denseType A_P_alpha;
    get_same_shape_denseType(B, &A_P_alpha);

    denseType P_beta;
    get_same_shape_denseType(B, &P_beta);

    // A * X0
    spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);


    // R0=B-AX0           
    dense_mat_mat_add_TP(B, A_X, R, 1.0, -1.0, myid, numprocs);

    //P0 = R0
    dense_entry_copy(R, P);

    denseType Pt_A_P;
    generate_no_parallel_dense(&Pt_A_P, P.local_num_col, P.local_num_col);
    //    denseType Pt_A_P_inv;
    //    generate_no_parallel_dense(&Pt_A_P_inv, P.local_num_col, P.local_num_col);

    denseType Rt_R;
    generate_no_parallel_dense(&Rt_R, R.local_num_col, R.local_num_col);
    denseType Rnewt_Rnew;
    generate_no_parallel_dense(&Rnewt_Rnew, R.local_num_col, R.local_num_col);

    denseType ALPHA_mat;
    generate_no_parallel_dense(&ALPHA_mat, P.local_num_col, R.local_num_col);

    denseType BETA_mat;
    generate_no_parallel_dense(&BETA_mat, R.local_num_col, R.local_num_col);


    // R0**t * R0
    MatTranposeMatMul(&Rt_R, R, myid, numprocs);

    long max_iter = BCG_MAXITER;
    long bcg_v1_loop_idx;

    // start main loop
    //
    for (bcg_v1_loop_idx = 0; bcg_v1_loop_idx < max_iter; bcg_v1_loop_idx++) {
        // A_P = A*P
        spmm_csr_v1(mat, P, &A_P, B_global_shape_swap_zone, myid, numprocs);

        // to calculate PtAP, B_global_shape_swap_zone need to be reused.
        // so the later call should be called immediately after A_P calculation
        // PtAP
        // distributedMatTransposeLocalMatMul_v1(&Pt_A_P, A_P, B_global_shape_swap_zone, myid, numprocs);        
        distributedMatLTransposeMatRMul(&Pt_A_P, P, A_P, myid, numprocs);

        // to solve linear system PtAp * alpha = RtR to get alpha
        // Pt_A_P items are changed after this call
        LUsolver_v1(&ALPHA_mat, Pt_A_P, Rt_R, myid, numprocs);

        // P * alpha
        dense_distributedMatL_localMatR_Mul_v1(&P_alpha, P, ALPHA_mat, myid, numprocs);

        // X_new=X_old + P_alpha
        // wasted flops, bad function def
        dense_mat_mat_add_TP(X, P_alpha, X, 1.0, 1.0, myid, numprocs);

        // AP * alpha
        dense_distributedMatL_localMatR_Mul_v1(&A_P_alpha, A_P, ALPHA_mat, myid, numprocs);

        // check vector norm
        long n_col = 1;
        double R_norm2squre;
        norm2square_dist_denseMat_col_n(R, &R_norm2squre, n_col, myid, numprocs);
        if (myid == 0 && (bcg_v1_loop_idx % NUM_LOOP_PER_PRINT==0 || sqrt(R_norm2squre) < epsilon \
            || bcg_v1_loop_idx == (max_iter))) {
            printf("Loop: %d, R[%d]_norm=%30.30f\n", bcg_v1_loop_idx, n_col, sqrt(R_norm2squre));
        }
        if (sqrt(R_norm2squre) < epsilon) {
            return;
        }
        // R_new = R_old - AP_alpha
        dense_mat_mat_add_TP(R, A_P_alpha, R, 1.0, -1.0, myid, numprocs);

        // calculate R_new **t * R_new
        MatTranposeMatMul(&Rnewt_Rnew, R, myid, numprocs);
        // solve R_old**t * R_old * beta = R_new**t * R_new to for beta
        // Rt_R item values are changed after this LUsolver_v1 call
        LUsolver_v1(&BETA_mat, Rt_R, Rnewt_Rnew, myid, numprocs);

        if (Rt_R.data != 0) {
            free(Rt_R.data);
        } else {
            exit(0);
        }

        // exchange data zone
        Rt_R.data = (double*) Rnewt_Rnew.data;
        Rnewt_Rnew.data = (void*) 0;

        // P_beta= P * beta
        dense_distributedMatL_localMatR_Mul_v1(&P_beta, P, BETA_mat, myid, numprocs);
        // P_new = R_new + P_beta
        // wasted flops, bad function def
        dense_mat_mat_add_TP(R, P_beta, P, 1.0, 1.0, myid, numprocs);

    }

    // garbage collection
    free(B_global_shape_swap_zone);
}


/*
 * //#define BCG_qr_DEBUG_P_NEW
void bcg_Scalapack_qr(csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs) {

    long context, desc[9], R_global_row, R_global_col,
            R_block_row, R_block_col, rsrc, csrc, lld,
            info, mype, npe, myrow, mycol, nprow, npcol;

    long *map = (long*) calloc(numprocs, sizeof (long));
    long map_idx;
    for (map_idx = 0; map_idx < numprocs; map_idx++) {
        map[map_idx] = map_idx;
    }

    long ierr;
    denseType R;
    // set up blacs context based on the existing MPI context
    Cblacs_pinfo(&mype, &npe);
    Cblacs_get(-1, 0, &context);
    nprow = npe;
    npcol = 1;
    Cblacs_gridmap(&context, map, nprow, nprow, npcol);
    Cblacs_gridinfo( context, &nprow, &npcol, &myrow, &mycol );
    printf ("myid: %d, myrow: %d, mycol: %d\n", myid, myrow, mycol);

    get_same_shape_denseType(B, &R);

    double * B_global_shape_swap_zone = (double*) calloc(B.global_num_col * B.global_num_row, sizeof (double));

    denseType A_X;
    get_same_shape_denseType(B, &A_X);
    //    set_dense_to_zero(A_X);

    denseType P;
    get_same_shape_denseType(B, &P);

    denseType A_P;
    get_same_shape_denseType(B, &A_P);

    denseType P_alpha;
    get_same_shape_denseType(B, &P_alpha);

    denseType A_P_alpha;
    get_same_shape_denseType(B, &A_P_alpha);

    denseType P_beta;
    get_same_shape_denseType(B, &P_beta);

    // A * X0
    spmm_csr_v1(mat, X, &A_X, B_global_shape_swap_zone, myid, numprocs);

    // R0=B-AX0           
    dense_mat_mat_add_TP(B, A_X, R, 1.0, -1.0, myid, numprocs);

    //    //P0 = R0
    //    dense_entry_copy(R, P);
    // Ro = qgamma, P0 = q
    dense_entry_copy(R, P);
    R_global_row = R.global_num_row;
    R_global_col = R.global_num_col;
    R_block_row = R.local_num_row;
    R_block_col = R.local_num_col;
    rsrc = 0;
    csrc = 0;
//    lld = R.local_num_col; // now test QR fac of R**T
    lld = numroc_(&R_global_row , &R_block_row, &myrow, &rsrc, &nprow);
    descinit_(desc, &R_global_row, &R_global_col, &R_block_row, &R_block_col
            , &rsrc, &csrc, &context, &lld, &info);
    long idxt;
    for (idxt = 0; idxt < 9; idxt++) {
        if (myid == numprocs-1)
            printf("desc %d: %d\n", idxt, desc[idxt]);
    }
//    printf("info value:%d\n",info);
    printf("myid:%d, myrow:%d, mycol:%d, lld:%d\n",myid,myrow,mycol,lld);
    exit(0);

    denseType Pt_A_P;
    generate_no_parallel_dense(&Pt_A_P, P.local_num_col, P.local_num_col);
    //    denseType Pt_A_P_inv;
    //    generate_no_parallel_dense(&Pt_A_P_inv, P.local_num_col, P.local_num_col);

    denseType Rt_R;
    generate_no_parallel_dense(&Rt_R, R.local_num_col, R.local_num_col);
    denseType Rnewt_Rnew;
    generate_no_parallel_dense(&Rnewt_Rnew, R.local_num_col, R.local_num_col);

    denseType ALPHA_mat;
    generate_no_parallel_dense(&ALPHA_mat, P.local_num_col, R.local_num_col);

    denseType BETA_mat;
    generate_no_parallel_dense(&BETA_mat, R.local_num_col, R.local_num_col);


    // R0**t * R0
    MatTranposeMatMul(&Rt_R, R, myid, numprocs);

    long max_iter = 1000;
    long bcg_v1_loop_idx;
    for (bcg_v1_loop_idx = 0; bcg_v1_loop_idx < max_iter; bcg_v1_loop_idx++) {
        // A_P = A*P
        spmm_csr_v1(mat, P, &A_P, B_global_shape_swap_zone, myid, numprocs);

        // to calculate PtAP, B_global_shape_swap_zone need to be reused.
        // so the later call should be called immediately after A_P calculation
        // PtAP
        // distributedMatTransposeLocalMatMul_v1(&Pt_A_P, A_P, B_global_shape_swap_zone, myid, numprocs);        
        distributedMatLTransposeMatRMul(&Pt_A_P, P, A_P, myid, numprocs);

        // to solve linear system PtAp * alpha = RtR to get alpha
        // Pt_A_P items are changed after this call
        LUsolver_v1(&ALPHA_mat, Pt_A_P, Rt_R, myid, numprocs);

        // P * alpha
        dense_distributedMatL_localMatR_Mul_v1(&P_alpha, P, ALPHA_mat, myid, numprocs);

        // X_new=X_old + P_alpha
        // wasted flops, bad function def
        dense_mat_mat_add_TP(X, P_alpha, X, 1.0, 1.0, myid, numprocs);

        // AP * alpha
        dense_distributedMatL_localMatR_Mul_v1(&A_P_alpha, A_P, ALPHA_mat, myid, numprocs);

        // check vector norm
        long n_col = 1;
        double R_norm2squre;
        norm2square_dist_denseMat_col_n(R, &R_norm2squre, n_col, myid, numprocs);
        if (myid == 0) {
            printf("Loop: %d, R[%d]_norm=%30.30f\n", bcg_v1_loop_idx, n_col, sqrt(R_norm2squre));
        }
        if (sqrt(R_norm2squre) < epsilon) {
            return;
        }
        // R_new = R_old - AP_alpha
        dense_mat_mat_add_TP(R, A_P_alpha, R, 1.0, -1.0, myid, numprocs);

        // calculate R_new **t * R_new
        MatTranposeMatMul(&Rnewt_Rnew, R, myid, numprocs);
        // solve R_old**t * R_old * beta = R_new**t * R_new to for beta
        // Rt_R item values are changed after this LUsolver_v1 call
        LUsolver_v1(&BETA_mat, Rt_R, Rnewt_Rnew, myid, numprocs);

        if (Rt_R.data != 0) {
            free(Rt_R.data);
        } else {
            exit(0);
        }

        // exchange data zone
        Rt_R.data = (double*) Rnewt_Rnew.data;
        Rnewt_Rnew.data = (void*) 0;

        // P_beta= P * beta
        dense_distributedMatL_localMatR_Mul_v1(&P_beta, P, BETA_mat, myid, numprocs);
        // P_new = R_new + P_beta
        // wasted flops, bad function def
        dense_mat_mat_add_TP(R, P_beta, P, 1.0, 1.0, myid, numprocs);
#ifdef BCG_qr_DEBUG_P_NEW
        if (myid == 0 && bcg_v1_loop_idx == 5) {
            printf("In loop %d, P_new elements are: \n", bcg_v1_loop_idx);
            local_dense_mat_print(P, myid);
        }
#endif

    }

    // garbage collection
    free(B_global_shape_swap_zone);
}
 */
