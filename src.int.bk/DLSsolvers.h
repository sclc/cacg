/* 
 * File:   DLSsolvers.h
 * Author: sc2012
 *
 * Created on January 4, 2014, 3:17 AM
 */
#include "DataTypes.h"
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "MatrixOperations.h"

#ifndef DLSSOLVERS_H
#define	DLSSOLVERS_H

#ifdef	__cplusplus
extern "C" {
#endif

    //  lhsMat * resMat = rhsMat
    //  lhsMat values will be changed, and LU decomposition results will be in lhsMat
    void LUsolver_v1 (denseType *resMat, denseType lhsMat, denseType rhsMat,int myid, int numprocs);
    
    //  lhsMat * resMat = rhsMat
    //  lhsMat values will not be changed, and LU decomposition results will not be saved
    void LUsolver_v1_KeeplhsMat (denseType *resMat, denseType lhsMatdb, denseType rhsMat,int myid, int numprocs);


#ifdef	__cplusplus
}
#endif

#endif	/* DLSSOLVERS_H */

