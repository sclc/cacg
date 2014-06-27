/* 
 * File:   GetRHS.h
 * Author: scl
 *
 * Created on December 11, 2013, 9:14 PM
 */


#ifdef	__cplusplus
extern "C" {
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "DataTypes.h"
#include "common.h"
    
void GenVectorOne(int length, denseType * vector, int num_cols,int myid, int numprocs);

// 1st: make sure csv file has enough elements, dense matrix size is given
// 2nd: rank0 read matrix, and then distribute MPI_Scatter

void GenVector_ReadCSV(denseType * vector, int length, int num_cols, char* rhsFile,int myid, int numprocs);


#ifdef	__cplusplus
}
#endif


