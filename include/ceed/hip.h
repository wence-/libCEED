#ifndef _ceed_hip_h
#define _ceed_hip_h

#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

CEED_EXTERN int CeedQFunctionSetHIPUserFunction(CeedQFunction qf,
    hipFunction_t f);

#endif
