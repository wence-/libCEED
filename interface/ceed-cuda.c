// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
#include <ceed/cuda.h>
#include <ceed-impl.h>

/**
  @brief Set CUDA function pointer to evaluate action at quadrature points

  @param qf  CeedQFunction to set device pointer
  @param f   Device function pointer to evaluate action at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f) {
  int ierr;
  if (!qf->SetCUDAUserFunction) {
    Ceed ceed;
    ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
    CeedDebug(ceed, "Backend does not support CUfunction pointers for QFunctions.");
  } else {
    ierr = qf->SetCUDAUserFunction(qf, f); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}
