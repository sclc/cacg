#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

#include "DataTypes.h"
char * concatStr (char * s1, char * s2);

void parseCSV (char* filename, double** output, int numRows, int numCols);

void gerschgorin_v1 (double * output, csrType_local mat);

void CBCG_chebyshevPolynomialBasisGen (denseType S_mat,  csrType_local mat, denseType R, int s
                                , double s_alpha, double s_beta, int myid, int numprocs);

void BCBCG_chebyshevPolynomialBasisGen (denseType S_mat,  csrType_local mat, denseType R, int s
                                , double s_alpha, double s_beta, int myid, int numprocs);
