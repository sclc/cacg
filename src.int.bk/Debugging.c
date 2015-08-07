#include "Debugging.h"

void check_coo_matrix_print(cooType mat, matInfo *info)
{
    int idx;
    for (idx = 0; idx<info->nnz; idx++)
        printf ("%d %d %10.10f\n", mat.rowIdx[idx] + 1, mat.colIdx[idx] + 1, mat.coodata[idx]);
}

void check_csr_matrix_print(csrType_local mat)
{
    int row_idx, val_idx;
    int nnz_counter = 0;
    for (row_idx=0; row_idx < mat.num_rows; row_idx++)
    {
        int start = mat.row_start[row_idx];
        int end   = mat.row_start[row_idx + 1];
        
        for (val_idx = start; val_idx<end; val_idx++)
            printf ("row: %d, col= %d, val= %10.10lf\n", row_idx, mat.col_idx[val_idx], mat.csrdata[val_idx]);
        
        nnz_counter += (end - start);
        
    }
    
    assert(nnz_counter == mat.nnz);
   
}

void check_csv_array_print(double* array, int rows, int cols, int myid){
    int rowCounter;
    int colCounter;
    
    printf ("myid: %d, Printing %dX%d matrix", myid, rows, cols);
    for (rowCounter=0; rowCounter<rows; rowCounter++){
        printf("\n");
        printf("%f", array[rowCounter*cols]);
        for(colCounter=1; colCounter<cols; colCounter++){
            printf(", %f", array[rowCounter*cols + colCounter]); 
        }
    }
    printf("\ncheck_csv_array_print Printing done\n");
}