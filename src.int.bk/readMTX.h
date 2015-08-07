#include "DataTypes.h"
#include "mmio.h"
#include <assert.h>



void readMtx_coo(char* path, char* name, cooType mtr, matInfo info);
void readMtx_info_and_coo(char* path, char* name, matInfo* info, cooType* mat);
