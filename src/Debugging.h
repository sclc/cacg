/* 
 * File:   Debugging.h
 * Author: scl
 *
 * Created on November 27, 2013, 6:51 PM
 */
#include "DataTypes.h"
#include <stdio.h>
#include <assert.h>


#ifdef	__cplusplus
extern "C" {
#endif

    void check_coo_matrix_print(cooType mat, matInfo *info);
    void check_csr_matrix_print(csrType_local mat);
    void check_csv_array_print(double* array, int rows, int cols, int myid);

#ifdef	__cplusplus
}
#endif


