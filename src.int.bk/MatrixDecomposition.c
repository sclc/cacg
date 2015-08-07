#include "MatrixDecomposition.h"

void HoseHolder_QR_seq_v1(denseType mat, denseType q, denseType r) {
    assert(mat.local_num_row > mat.global_num_col);
    if (q.data != 0) {
        free(q.data);
    }
    int local_row_num = mat.local_num_row;
    int local_col_num = mat.local_num_col;
    double * colDataTemp = (double*) calloc(local_row_num, sizeof (double));

    //ldmR: leading dimension
    int ldmR = local_col_num;
    // partial QR
    int ldmQ = local_col_num;
    q.data = (double*) calloc(local_row_num * local_col_num, sizeof (double));

    // initialize q
    int idx;
    for (idx = 0; idx < local_col_num; idx++) {
        q.data[idx * ldmQ + idx] = 1.0;
    }

    int stepsTotal = local_col_num;
    int stepIdx;
    for (stepIdx = 0; stepIdx < stepsTotal; stepIdx++) {
        double maxColVal = 0.0;

        for (idx = stepIdx; idx < local_row_num; idx++) {
            double temp = mat.data[idx * ldmR + stepIdx];
            colDataTemp[idx] = temp;
            temp = fabs(temp);
            if (temp > maxColVal) {
                maxColVal = temp;
            }
        }

        double alpha = 0.0;
        for (idx = stepIdx; idx < local_row_num; idx++) {
            double temp = colDataTemp[idx] / maxColVal;
            alpha += temp * temp;            
        }
        if (colDataTemp[stepIdx] > 0.0){
            maxColVal = -maxColVal;
        }
        
        alpha = maxColVal * sqrt(alpha);
        
        // strange code
        if(fabs(alpha) + 1.0 == 1.0){
            printf("qr failed\n");
            exit(1);
        }
        
        double fac = sqrt(2.0 * alpha * ( alpha- colDataTemp[stepIdx]) );
        assert ( (fac+1.0) != 1.0);
        fac = 1.0 /fac;
        // calculate householder vector
        colDataTemp[stepIdx] = (colDataTemp[stepIdx] - alpha) * fac;       
        for (idx=stepIdx+1; idx < local_row_num; idx++) {
            colDataTemp[idx] *= fac;
        }
        
        // update q
        int colIdx, rowIdx;
        double totalTemp;
        for (colIdx = 0; colIdx < local_col_num; colIdx++){
            totalTemp = 0.0;
            for (rowIdx=stepIdx; rowIdx<local_row_num; rowIdx++){
                totalTemp += colDataTemp[rowIdx] * q.data[ rowIdx*ldmQ + colIdx];
            }
            for (rowIdx=stepIdx; rowIdx<local_row_num; rowIdx++){
                q.data[rowIdx*ldmQ + colIdx] -= 2.0 * colDataTemp[rowIdx] * totalTemp;
            }
        }
        
        // update mat
        
    }




}

void HouseHolder_TSQR_v1(denseType mat, denseType q, denseType r) {
    assert(mat.local_num_row > mat.global_num_col);

}

void GramSchmidt_QR_v1(denseType mat, denseType q, denseType r) {

}