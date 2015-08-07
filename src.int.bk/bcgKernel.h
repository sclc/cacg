#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "DataTypes.h"

//extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
//extern void   Cblacs_get( int context, int request, int* value);
//extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
//extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
//extern void   Cblacs_gridexit( int context);
//extern void   Cblacs_exit( int error_code);
////still not know the prototype for this function
//extern void Cblacs_gridmap();
//
//extern void   descinit_( int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc,
//            int *ictxt, int *lld, int *info);
//extern int numroc_();

void bcg_v1 (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);
//qr
void bcg_QR (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);

//
//void bcg_Scalapack_qr (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);