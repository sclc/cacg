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

	long num_rows;
	long num_cols;
	long nnz;

} matInfo;

typedef struct
{
                 
    /* Dimensions, number of nonzeros 
     * (m == n for square, m < n for a local "slice") */
    // local matrix property
    long num_rows, num_cols, nnz;
                     
    /* Start of the rows owned by each thread */
    long start;      

    /* Starts of rows owned by local processor
     * row_start[0] == 0 == offset of first nz in row start[MY_THREAD] 
     */
    long *row_start;
                                          
    /* Column indices and values of matrix elements at local processor */
    long *col_idx;
    double *csrdata;
                                          
} csrType_local;

typedef struct 
{
	long * rowIdx;
	long * colIdx;
	double * coodata;
} cooType;

typedef struct
{
	long local_num_row;
	long local_num_col;

    long global_num_row;
    long global_num_col;
    long start_idx;
        
	double * data;
} denseType;


#ifdef	__cplusplus
}
#endif

#endif	/* DATATYPES_H */

