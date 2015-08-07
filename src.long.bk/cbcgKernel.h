/* 
 * File:   cbcgKernel.h
 * Author: sc2012
 *
 * Created on January 12, 2014, 9:53 AM
 */

#ifndef CBCGKERNEL_H
#define	CBCGKERNEL_H

#ifdef	__cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
    
#include "common.h"
#include "DataTypes.h"
#include "DLSsolvers.h"
    
void cbcg_v1 (csrType_local mat, denseType B, denseType X, long s, double epsilon, int myid, int numprocs);


#ifdef	__cplusplus
}
#endif

#endif	/* CBCGKERNEL_H */

