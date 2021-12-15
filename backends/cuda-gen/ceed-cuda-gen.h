#ifndef _ceed_cuda_gen_h
#define _ceed_cuda_gen_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include "../cuda/ceed-cuda.h"

typedef struct { const CeedScalar *in[CEED_FIELD_MAX]; CeedScalar *out[CEED_FIELD_MAX]; } CudaFields;
typedef struct { CeedInt *in[CEED_FIELD_MAX]; CeedInt *out[CEED_FIELD_MAX]; } CudaFieldsInt;

typedef struct {
  CeedInt dim;
  CeedInt Q1d;
  CeedInt maxP1d;
  CUmodule module;
  CUfunction op;
  CudaFieldsInt indices;
  CudaFields fields;
  CudaFields B;
  CudaFields G;
  CeedScalar *W;
} CeedOperator_Cuda_gen;

typedef struct {
  char *qFunctionName;
  char *qFunctionSource;
  void *d_c;
} CeedQFunction_Cuda_gen;

CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Cuda_gen(CeedOperator op);

#endif // _ceed_cuda_gen_h
