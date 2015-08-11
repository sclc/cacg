#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "DataTypes.h"

//extern void   Cblacs_pinfo( long* mypnum, long* nprocs);
//extern void   Cblacs_get( long context, long request, long* value);
//extern long    Cblacs_gridinit( long* context, char * order, long np_row, long np_col);
//extern void   Cblacs_gridinfo( long context, long*  np_row, long* np_col, long*  my_row, long*  my_col);
//extern void   Cblacs_gridexit( long context);
//extern void   Cblacs_exit( long error_code);
////still not know the prototype for this function
//extern void Cblacs_gridmap();
//
//extern void   descinit_( long *desc, long *m, long *n, long *mb, long *nb, long *irsrc, long *icsrc,
//            long *ictxt, long *lld, long *info);
//extern long numroc_();

void bcg_v1 (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);
//qr
void bcg_QR (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);

//
//void bcg_Scalapack_qr (csrType_local mat, denseType B, denseType X, double epsilon, int myid, int numprocs);