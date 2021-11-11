// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <arm_sve.h>
#include <stdbool.h>
#include "ceed-sve.h"

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sve(CeedTensorContract contract, CeedInt A,
                                       CeedInt B, CeedInt C, CeedInt J,
                                       const double *restrict t,
                                       CeedTransposeMode t_mode,
                                       const CeedInt add,
                                       const double *restrict u,
                                       double *restrict v) {
  const CeedInt blk_size = 8;

  if (!add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (double) 0.0;

  if (C == 1) {
    // Serial C=1 Case
    CeedTensorContract_Avx_Single_4_8(contract, A, B, C, J, t, t_mode, true, u,
                                      v);
  } else {
    // Blocks of 8 columns
    if (C >= blk_size)
      CeedTensorContract_Avx_Blocked_4_8(contract, A, B, C, J, t, t_mode, true,
                                         u, v);
    // Remainder of columns
    if (C % blk_size)
      CeedTensorContract_Avx_Remainder_8_8(contract, A, B, C, J, t, t_mode, true,
                                           u, v);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Sve(CeedTensorContract contract) {
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f64_Sve(CeedBasis basis,
                                     CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Sve); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Sve); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
