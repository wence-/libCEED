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

#ifndef _ceed_cuda_shared_h
#define _ceed_cuda_shared_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include "../cuda/ceed-cuda-common.h"

typedef struct {
  CUmodule module;
  CUfunction Interp;
  CUfunction Grad;
  CUfunction Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_collo_grad_1d;
  CeedScalar *d_q_weight_1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
} CeedBasis_Cuda_shared;

CEED_INTERN int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d,
    CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

#endif // _ceed_cuda_shared_h
