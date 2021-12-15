#include <ceed/ceed.h>
#include <ceed/backend.h>
#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Tensor Contract Core loop
//------------------------------------------------------------------------------
static inline int CeedTensorContractApply_Core_Opt(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1; t_stride_1 = J;
  }

  for (CeedInt a=0; a<A; a++)
    for (CeedInt b=0; b<B; b++)
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*t_stride_0 + b*t_stride_1];
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Opt(CeedTensorContract contract, CeedInt A,
                                       CeedInt B, CeedInt C, CeedInt J,
                                       const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add,
                                       const CeedScalar *restrict u,
                                       CeedScalar *restrict v) {
  if (!add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  if (C == 1)
    return CeedTensorContractApply_Core_Opt(contract, A, B, 1, J, t, t_mode,
                                            add, u, v);
  else
    return CeedTensorContractApply_Core_Opt(contract, A, B, C, J, t, t_mode,
                                            add, u, v);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Opt(CeedTensorContract contract) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Opt(CeedBasis basis, CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Opt); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Opt); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
