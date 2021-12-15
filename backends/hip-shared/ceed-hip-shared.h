#ifndef _ceed_hip_shared_h
#define _ceed_hip_shared_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include "../hip/ceed-hip.h"

typedef struct {
  hipModule_t module;
  hipFunction_t interp;
  hipFunction_t grad;
  hipFunction_t weight;
  CeedInt blksizes[3]; // interp, grad, weight thread block sizes
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
  CeedScalar *d_collograd1d;
  CeedScalar *d_qweight1d;
} CeedBasis_Hip_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Hip_shared(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
    const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

#endif // _ceed_hip_shared_h
