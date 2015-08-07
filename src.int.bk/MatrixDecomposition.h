/* 
 * File:   MatrixDecomposition.h
 * Author: scl
 *
 * Created on January 24, 2014, 4:33 PM
 */

#ifndef MATRIXDECOMPOSITION_H
#define	MATRIXDECOMPOSITION_H

#ifdef	__cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "DataTypes.h"

    // a sequental vesion of qr factorization
    // now, assuming mat.row_num/numprocs > mat.col_num, for we are 
    // handling tall-skinny matrix QR decomposition
    void HoseHolder_QR_seq_v1(denseType mat, denseType q, denseType r);
    
    void HouseHolder_TSQR_v1(denseType mat, denseType q, denseType r);

    void GramSchmidt_QR_v1(denseType mat, denseType q, denseType r);


#ifdef	__cplusplus
}
#endif

#endif	/* MATRIXDECOMPOSITION_H */

