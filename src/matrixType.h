// define matrix types
#include "DataTypes.h"
#include <stdlib.h>

void Converter_Coo2Csr (cooType src, csrType_local * target, matInfo * mat_info);

void delete_csrType_local   (csrType_local mat);
void delete_cooType   (cooType mat);
void delete_denseType (denseType mat);

void get_same_shape_denseType (denseType src, denseType *target);
void set_dense_to_zero (denseType mat);

void gen_dense_mat (denseType *mat, int local_row_num, int local_col_num
                  , int global_row_num, int global_col_num, int start_row_id);