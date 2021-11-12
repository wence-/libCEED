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
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <stdbool.h>
#include "ceed-sve.h"

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sve_Blocked(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
    CeedTransposeMode t_mode, const CeedInt add, const double *restrict u,
    double *restrict v, const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1; t_stride_1 = J;
  }

  for (CeedInt a=0; a<A; a++)
    for (CeedInt b=0; b<B; b++)
      // Blocks of JJ rows
      for (CeedInt j=0; j<(J/JJ)*JJ; j+=JJ)
        for (CeedInt jj=0; jj<JJ; jj++) { // unroll
          // C vectorization by compiler
          int32_t i = (a*J+j+jj)*C;
          svbool_t pg = svwhilelt_b64(i, C);
          do {
            // Load u, v into vectors
            svfloat64_t u_vec = svld1(pg, &u[i]);
            svfloat64_t v_vec = svld1(pg, &v[i]);
            // Basis matrix value
            double tq = t[(j+jj)*t_stride_0 + b*t_stride_1];
            // fmadd
            svst1(pg, &v[i], svmla_x(pg, v_vec, u_vec, tq));
            // Loop update
            i += svcntd();
            pg = svwhilelt_b64(i, C);
          } while (svptest_any(svptrue_b64(), pg));
        }
  // Remainder of rows
  CeedInt j=(J/JJ)*JJ;
  if (j < J) {
    for (CeedInt a=0; a<A; a++)
      for (CeedInt b=0; b<B; b++)
        // Blocks of JJ rows
        for (CeedInt jj=0; jj<J-j; jj++) { // not unrolled
          // C vectorization by compiler
          int32_t i = (a*J+j+jj)*C;
          svbool_t pg = svwhilelt_b64(i, C);
          do {
            // Load u, v into vectors
            svfloat64_t u_vec = svld1(pg, &u[i]);
            svfloat64_t v_vec = svld1(pg, &v[i]);
            // Basis matrix value
            double tq = t[(j+jj)*t_stride_0 + b*t_stride_1];
            // fmadd
            svst1(pg, &v[i], svmla_x(pg, v_vec, u_vec, tq));
            // Loop update
            i += svcntd();
            pg = svwhilelt_b64(i, C);
          } while (svptest_any(svptrue_b64(), pg));
        }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Serial Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sve_Single(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
    CeedTransposeMode t_mode, const CeedInt add, const double *restrict u,
    double *restrict v, const CeedInt AA, const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1; t_stride_1 = J;
  }

  // Blocks of A rows
  for (CeedInt a=0; a<(A/AA)*AA; a+=AA) {
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt aa=0; aa<AA; aa++) {
          for (CeedInt jj=0; jj<JJ/4; jj++) { // unroll
            // A vectorization by compiler
            int32_t j = 0;
            svbool_t pg = svwhilelt_b64(j, (J/JJ)*JJ);
            do {
              // Load u, v into vectors
              svfloat64_t u_vec = svld1(pg, &u[(a+aa)*B+b]);
              svfloat64_t v_vec = svld1(pg, &v[(a+aa)*J+j]);
              svfloat64_t tq;
              // Basis matrix value
              double tqv[4] = {t[(j+jj*4+3)*t_stride_0 + b*t_stride_1],
                               t[(j+jj*4+2)*t_stride_0 + b*t_stride_1],
                               t[(j+jj*4+1)*t_stride_0 + b*t_stride_1],
                               t[(j+jj*4+0)*t_stride_0 + b*t_stride_1]};
              svst(pg, &tq[0], tqv);
              // fmadd
              svst(pg, &v[(a+aa)*J+j], svmla_x(pg, v_vec, u_vec, &tq));
              // Loop update
              j += svcntd();
              pg = svwhilelt_b64(j, (J/JJ)*JJ);
            } while (svptest_any(svptrue_b64(), pg));
          }
        }
      }
  }

  // Remainder of rows
  CeedInt a=(A/AA)*AA;
  for (CeedInt aa=0; aa<A-a; aa++) {
    for (CeedInt b=0; b<B; b++) {
      for (CeedInt jj=0; jj<JJ/4; jj++) { // unroll
        // A vectorization by compiler
        int32_t j = 0;
        svbool_t pg = svwhilelt_b64(j, (J/JJ)*JJ);
        do {
          // Load u, v into vectors
          // Load u, v into vectors
          svfloat64_t u_vec = svld1(pg, &u[(a+aa)*B+b]);
          svfloat64_t v_vec = svld1(pg, &v[(a+aa)*J+j]);
          svfloat64_t tq;
          // Basis matrix value
          double tqv[4] = {t[(j+jj*4+3)*t_stride_0 + b*t_stride_1],
                           t[(j+jj*4+2)*t_stride_0 + b*t_stride_1],
                           t[(j+jj*4+1)*t_stride_0 + b*t_stride_1],
                           t[(j+jj*4+0)*t_stride_0 + b*t_stride_1]};
          svst(pg, &tq[0], tqv);
          // fmadd
          svst(pg, &v[(a+aa)*J+j], svmla_x(pg, v_vec, u_vec, tq));
          // Loop update
          j += svcntd();
          pg = svwhilelt_b64(j, (J/JJ)*JJ);
        } while (svptest_any(svptrue_b64(), pg));
      }
    }
  }

  // Column remainder
  CeedInt A_break = A%AA ? (A/AA)*AA : (A/AA-1)*AA;
  for (CeedInt j = (J/JJ)*JJ; j<J; j+=A) {
    // Blocks of A rows
    for (CeedInt a=0; a<A_break; a+=AA) {
      for (CeedInt aa=0; aa<AA; aa++) {
        for (CeedInt b=0; b<B; b++) {
            int32_t j = (J/JJ)*JJ;
            svbool_t pg = svwhilelt_b64(j, J);
            do {
              svfloat64_t tq;
              if (J-j == 1) {
                double tqv[4] = {0.0, 0.0, 0.0, t[(j+0)*t_stride_0 + b*t_stride_1]};
                svst(pg, &tq[0], tqv);
              } else if (J-j == 2) {
                double tqv[4] = {0.0, 0.0, t[(j+1)*t_stride_0 + b*t_stride_1],
                                 t[(j+0)*t_stride_0 + b*t_stride_1]};
                svst(pg, &tq[0], tqv);
              } else if (J-3 == j) {
                double tqv[4] = {0.0, t[(j+2)*t_stride_0 + b*t_stride_1],
                                 t[(j+1)*t_stride_0 + b*t_stride_1],
                                 t[(j+0)*t_stride_0 + b*t_stride_1]};
                svst(pg, &tq[0], tqv);
              } else {
                double tqv[4] = {t[(j+3)*t_stride_0 + b*t_stride_1],
                                 t[(j+2)*t_stride_0 + b*t_stride_1],
                                 t[(j+1)*t_stride_0 + b*t_stride_1],
                                 t[(j+0)*t_stride_0 + b*t_stride_1]};
                svst(pg, &tq[0], tqv);
              }
              // Load u, v into vectors
              svfloat64_t u_vec = svld1(pg, &u[(a+aa)*B+b]);
              svfloat64_t v_vec = svld1(pg, &v[(a+aa)*J+j]);

              // fmadd
              svst(pg, &v[(a+aa)*J+j], svmla_x(pg, v_vec, u_vec, tq));
              // Loop update
              j += svcntd();
              pg = svwhilelt_b64(j, J);
            } while (svptest_any(svptrue_b64(), pg));
        }
      }
    }
  }
  // Remainder of rows, all columns
  for (CeedInt b=0; b<B; b++) {
    for (CeedInt j=(J/JJ)*JJ; j<J; j++) {
      double tq = t[j*t_stride_0 + b*t_stride_1];
      for (CeedInt a=A_break; a<A; a++)
        v[a*J+j] += tq * u[a*B+b];
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract - Common Sizes
//------------------------------------------------------------------------------
static int CeedTensorContract_Sve_Blocked_4_8(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
    CeedTransposeMode t_mode, const CeedInt add, const double *restrict u,
    double *restrict v) {
  return CeedTensorContract_Sve_Blocked(contract, A, B, C, J, t, t_mode, add, u,
                                        v, 8);
}

static int CeedTensorContract_Sve_Single_4_8(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
    CeedTransposeMode t_mode, const CeedInt add, const double *restrict u,
    double *restrict v) {
  return CeedTensorContract_Sve_Single(contract, A, B, C, J, t, t_mode, add, u,
                                       v, 4, 8);
}

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
    CeedTensorContract_Sve_Single_4_8(contract, A, B, C, J, t, t_mode, true, u,
                                      v);
  } else {
    // Blocks of 8 columns
      CeedTensorContract_Sve_Blocked_4_8(contract, A, B, C, J, t, t_mode, true,
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
