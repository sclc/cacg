/* 
 * File:   DataTypes.h
 * Author: scl
 *
 * Created on November 27, 2013, 5:29 PM
 */

#ifndef DATATYPES_H
#define	DATATYPES_H

#ifdef	__cplusplus
extern "C" {
#endif
typedef struct {

	int num_rows;
	int num_cols;
	int nnz;

} matInfo;

typedef struct
{
                 
    /* Dimensions, number of nonzeros 
     * (m == n for square, m < n for a local "slice") */
    // local matrix property
    int num_rows, num_cols, nnz;
                     
    /* Start of the rows owned by each thread */
    int start;      

    /* Starts of rows owned by local processor
     * row_start[0] == 0 == offset of first nz in row start[MY_THREAD] 
     */
    int *row_start;
                                          
    /* Column indices and values of matrix elements at local processor */
    int *col_idx;
    double *csrdata;
                                          
} csrType_local;

typedef struct 
{
	int * rowIdx;
	int * colIdx;
	double * coodata;
} cooType;

typedef struct
{
	int local_num_row;
	int local_num_col;
        int global_num_row;
        int global_num_col;
        int start_idx;
        
	double * data;
} denseType;


#ifdef	__cplusplus
}
#endif

#endif	/* DATATYPES_H */

