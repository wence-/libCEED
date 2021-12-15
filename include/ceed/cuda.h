#ifndef _ceed_cuda_h
#define _ceed_cuda_h

#include <ceed/ceed.h>
#include <cuda.h>

CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf,
    CUfunction f);

#endif
