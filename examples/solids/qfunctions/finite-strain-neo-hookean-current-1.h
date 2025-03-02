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

/// @file
/// Hyperelasticity, finite strain for solid mechanics example using PETSc

#ifndef ELAS_FSCurrentNH1_H
#define ELAS_FSCurrentNH1_H

#include <math.h>

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Series approximation of log1p()
//  log1p() is not vectorized in libc
//
//  The series expansion is accurate to 1e-7 in the range sqrt(2)/2 < J < sqrt(2),
//  with machine precision accuracy near J=1.  The initialization extends this range
//  to 0.35 ~= sqrt(2)/4 < J < sqrt(2)*2 ~= 2.83, which should be sufficient for
//  applications of the Neo-Hookean model.
// -----------------------------------------------------------------------------
#ifndef LOG1P_SERIES_SHIFTED
#define LOG1P_SERIES_SHIFTED
CEED_QFUNCTION_HELPER CeedScalar log1p_series_shifted(CeedScalar x) {
  const CeedScalar left = sqrt(2.)/2 - 1, right = sqrt(2.) - 1;
  CeedScalar sum = 0;
  if (1) { // Disable if the smaller range sqrt(2)/2 < J < sqrt(2) is sufficient
    if (x < left) { // Replace if with while for arbitrary range (may hurt vectorization)
      sum -= log(2.) / 2;
      x = 1 + 2 * x;
    } else if (right < x) {
      sum += log(2.) / 2;
      x = (x - 1) / 2;
    }
  }
  CeedScalar y = x / (2. + x);
  const CeedScalar y2 = y*y;
  sum += y;
  y *= y2;
  sum += y / 3;
  y *= y2;
  sum += y / 5;
  y *= y2;
  sum += y / 7;
  return 2 * sum;
};
#endif

// -----------------------------------------------------------------------------
// Compute det F - 1
// -----------------------------------------------------------------------------
#ifndef DETJM1
#define DETJM1
CEED_QFUNCTION_HELPER CeedScalar computeJM1(const CeedScalar grad_u[3][3]) {
  return grad_u[0][0]*(grad_u[1][1]*grad_u[2][2]-grad_u[1][2]*grad_u[2][1]) +
         grad_u[0][1]*(grad_u[1][2]*grad_u[2][0]-grad_u[1][0]*grad_u[2][2]) +
         grad_u[0][2]*(grad_u[1][0]*grad_u[2][1]-grad_u[2][0]*grad_u[1][1]) +
         grad_u[0][0] + grad_u[1][1] + grad_u[2][2] +
         grad_u[0][0]*grad_u[1][1] + grad_u[0][0]*grad_u[2][2] +
         grad_u[1][1]*grad_u[2][2] - grad_u[0][1]*grad_u[1][0] -
         grad_u[0][2]*grad_u[2][0] - grad_u[1][2]*grad_u[2][1];
};
#endif

// -----------------------------------------------------------------------------
// Compute matrix^(-1), where matrix is nonsymetric, returns array of 9
// -----------------------------------------------------------------------------
#ifndef MatinvNonSym
#define MatinvNonSym
CEED_QFUNCTION_HELPER int computeMatinvNonSym(const CeedScalar A[3][3],
    const CeedScalar detA, CeedScalar Ainv[9]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[9] = {A[1][1]*A[2][2] - A[1][2]*A[2][1], /* *NOPAD* */
                     A[0][0]*A[2][2] - A[0][2]*A[2][0], /* *NOPAD* */
                     A[0][0]*A[1][1] - A[0][1]*A[1][0], /* *NOPAD* */
                     A[0][2]*A[1][0] - A[0][0]*A[1][2], /* *NOPAD* */
                     A[0][1]*A[1][2] - A[0][2]*A[1][1], /* *NOPAD* */
                     A[0][2]*A[2][1] - A[0][1]*A[2][2], /* *NOPAD* */
                     A[0][1]*A[2][0] - A[0][0]*A[2][1], /* *NOPAD* */
                     A[1][0]*A[2][1] - A[1][1]*A[2][0], /* *NOPAD* */
                     A[1][2]*A[2][0] - A[1][0]*A[2][2]  /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 9; m++)
    Ainv[m] = B[m] / (detA);

  return 0;
};
#endif

// -----------------------------------------------------------------------------
// Common computations between Ftau and dFtau
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int commonFtau(const CeedScalar lambda,
                                     const CeedScalar mu, const CeedScalar Grad_u[3][3],
                                     CeedScalar Finv[3][3], CeedScalar tau_work[6],
                                     CeedScalar *llnj) {


  // Compute The Deformation Gradient : F = I3 + Grad_u
  // *INDENT-OFF*
  const CeedScalar F[3][3] =  {{Grad_u[0][0] + 1,
                                Grad_u[0][1],
                                Grad_u[0][2]},
                               {Grad_u[1][0],
                                Grad_u[1][1] + 1,
                                Grad_u[1][2]},
                               {Grad_u[2][0],
                                Grad_u[2][1],
                                Grad_u[2][2] + 1}
                              };

  // *INDENT-ON*
  //b - I3 = (Grad_u + Grad_u^T + Grad_u*Grad_u^T)
  const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  CeedScalar bMI3[6];
  for (CeedInt m = 0; m < 6; m++) {
    bMI3[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
    for (CeedInt n = 0; n < 3; n++)
      bMI3[m] += Grad_u[indj[m]][n] * Grad_u[indk[m]][n];
  }
  const CeedScalar Jm1 = computeJM1(Grad_u);
  const CeedScalar logJ = log1p_series_shifted(Jm1);

  //Computer F^(-1)
  const CeedScalar detF = Jm1 + 1.;
  CeedScalar Finvwork[9];
  computeMatinvNonSym(F, detF, Finvwork);

  Finv[0][0] = Finvwork[0];
  Finv[0][1] = Finvwork[5];
  Finv[0][2] = Finvwork[4];
  Finv[1][0] = Finvwork[8];
  Finv[1][1] = Finvwork[1];
  Finv[1][2] = Finvwork[3];
  Finv[2][0] = Finvwork[7];
  Finv[2][1] = Finvwork[6];
  Finv[2][2] = Finvwork[2];

  // Compute the Kirchhoff stress (tau) tau = mu*(b - I3) + lambda*log(J)*I3
  *llnj = lambda*logJ;

  tau_work[0] = mu*bMI3[0] + *llnj;
  tau_work[1] = mu*bMI3[1] + *llnj;
  tau_work[2] = mu*bMI3[2] + *llnj;
  tau_work[3] = mu*bMI3[3];
  tau_work[4] = mu*bMI3[4];
  tau_work[5] = mu*bMI3[5];

  return 0;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH1F)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*Grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  b     : left Cauchy-Green tensor
  //  binv  : inverse of b
  //  F     : deformation gradient
  //  tau   : Kirchhoff stress (in current config)
  // Formulation:
  //  F =  I3 + Grad_u
  //  J = det(F)
  //  b = F*F(^T)
  //  tau = mu*(b - I3) + lambda*log(J)*I3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar wdetJ      =   q_data[0][i];
    // dXdx_initial = dX/dx_initial
    // X is natural coordinate sys OR Reference [-1,1]^dim
    // x_initial is initial config coordinate system
    const CeedScalar dXdx_initial[3][3] = {{q_data[1][i],
                                            q_data[2][i],
                                            q_data[3][i]},
                                           {q_data[4][i],
                                            q_data[5][i],
                                            q_data[6][i]},
                                           {q_data[7][i],
                                            q_data[8][i],
                                            q_data[9][i]}
                                          };

    // *INDENT-ON*
    // Compute Grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        Grad_u[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          Grad_u[j][k][i] += du[j][m] * dXdx_initial[m][k];
      }

    // *INDENT-OFF*
    const CeedScalar tempGradu[3][3] =  {{Grad_u[0][0][i],
                                          Grad_u[0][1][i],
                                          Grad_u[0][2][i]},
                                         {Grad_u[1][0][i],
                                          Grad_u[1][1][i],
                                          Grad_u[1][2][i]},
                                         {Grad_u[2][0][i],
                                          Grad_u[2][1][i],
                                          Grad_u[2][2][i]}
                                        };
    // *INDENT-ON*

    // Common components of finite strain calculations
    CeedScalar Finv[3][3], tau_work[6], llnj;

    commonFtau(lambda, mu, tempGradu, Finv, tau_work, &llnj);
    // *INDENT-OFF*
    const CeedScalar tau[3][3] = {{tau_work[0], tau_work[5], tau_work[4]},
                                  {tau_work[5], tau_work[1], tau_work[3]},
                                  {tau_work[4], tau_work[3], tau_work[2]}
                                 };
    // x is current config coordinate system
    // dXdx = dX/dx = dX/dx_initial * F^{-1}
    // Note that F^{-1} = dx_initial/dx
    CeedScalar dXdx[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dXdx[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dXdx[j][k] += dXdx_initial[j][m] * Finv[m][k];
      }

    // Apply dXdx^T and weight to intermediate stress
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * tau[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH1dF)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // F is used for hyperelasticity (non-linear)
  const CeedScalar (*Grad_u)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of delta_u
    // *INDENT-OFF*
    const CeedScalar deltadu[3][3] = {{deltaug[0][0][i],
                                       deltaug[1][0][i],
                                       deltaug[2][0][i]},
                                      {deltaug[0][1][i],
                                       deltaug[1][1][i],
                                       deltaug[2][1][i]},
                                      {deltaug[0][2][i],
                                       deltaug[1][2][i],
                                       deltaug[2][2][i]}
                                     };
    // -- Qdata
    const CeedScalar wdetJ      =      q_data[0][i];
    // dXdx_initial = dX/dx_initial
    // X is natural coordinate sys OR Reference [-1,1]^dim
    // x_initial is initial config coordinate system
    const CeedScalar dXdx_initial[3][3] = {{q_data[1][i],
                                            q_data[2][i],
                                            q_data[3][i]},
                                           {q_data[4][i],
                                            q_data[5][i],
                                            q_data[6][i]},
                                           {q_data[7][i],
                                            q_data[8][i],
                                            q_data[9][i]}
                                          };
    // *INDENT-ON*

    // Compute Grad_du
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    // This is dF = Grad_du
    CeedScalar Grad_du[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        Grad_du[j][k] = 0;
        for (CeedInt m =0 ; m < 3; m++)
          Grad_du[j][k] += dXdx_initial[m][k] * deltadu[j][m];
      }

    // *INDENT-OFF*
    const CeedScalar tempGradu[3][3] =  {{Grad_u[0][0][i],
                                          Grad_u[0][1][i],
                                          Grad_u[0][2][i]},
                                         {Grad_u[1][0][i],
                                          Grad_u[1][1][i],
                                          Grad_u[1][2][i]},
                                         {Grad_u[2][0][i],
                                          Grad_u[2][1][i],
                                          Grad_u[2][2][i]}
                                        };
    // *INDENT-ON*

    // Common components of finite strain calculations
    CeedScalar F_inv[3][3], tau_work[6], llnj;

    // Common components of finite strain calculations (cur. config.)
    commonFtau(lambda, mu, tempGradu, F_inv, tau_work, &llnj);
    // *INDENT-OFF*
    const CeedScalar tau[3][3] = {{tau_work[0], tau_work[5], tau_work[4]},
                                  {tau_work[5], tau_work[1], tau_work[3]},
                                  {tau_work[4], tau_work[3], tau_work[2]}
                                 };

    // Compute grad_du = \nabla_x (deltau) = deltau * dX/dx
    CeedScalar grad_du[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        grad_du[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          grad_du[j][k] += deltadu[j][m]*F_inv[m][k];
      }

    // Compute grad_du_tau = grad_du*tau
    CeedScalar grad_du_tau[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        grad_du_tau[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          grad_du_tau[j][k] += grad_du[j][m]*tau[m][k];
      }

    // Compute depsilon = (grad_du + grad_du^T)/2
    const CeedScalar depsilon[3][3] = {{(grad_du[0][0] + grad_du[0][0])/2.,
                                        (grad_du[0][1] + grad_du[1][0])/2.,
                                        (grad_du[0][2] + grad_du[2][0])/2.},
                                       {(grad_du[1][0] + grad_du[0][1])/2.,
                                        (grad_du[1][1] + grad_du[1][1])/2.,
                                        (grad_du[1][2] + grad_du[2][1])/2.},
                                       {(grad_du[2][0] + grad_du[0][2])/2.,
                                        (grad_du[2][1] + grad_du[1][2])/2.,
                                        (grad_du[2][2] + grad_du[2][2])/2.}
                                      };
    // Compute trace(depsilon)
    CeedScalar tr_deps =  depsilon[0][0] + depsilon[1][1] + depsilon[2][2];
    // Compute grad_du*tau + trace(depsilon)I3
    grad_du_tau[0][0] += lambda*tr_deps;
    grad_du_tau[1][1] += lambda*tr_deps;
    grad_du_tau[2][2] += lambda*tr_deps;
    // Compute dp = grad_du*tau + trace(depsilon)I3 +2(mu-lambda*logJ)depsilon
    CeedScalar dp[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        dp[j][k] = grad_du_tau[j][k] + 2*(mu - llnj)*depsilon[j][k];
      }

    // x is current config coordinate system
    // dXdx = dX/dx = dX/dx_initial * F^{-1}
    // Note that F^{-1} = dx_initial/dx
    CeedScalar dXdx[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dXdx[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dXdx[j][k] += dXdx_initial[j][m] * F_inv[m][k];
      }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * dp[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH1Energy)(void *ctx, CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*energy) = (CeedScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar wdetJ      =   q_data[0][i];
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute Grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar Grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        Grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          Grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (Grad_u + Grad_u^T + Grad_u^T*Grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += Grad_u[n][indj[m]]*Grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*

    // Strain energy Phi(E) for compressible Neo-Hookean
    const CeedScalar Jm1 = computeJM1(Grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    energy[i] = (lambda*logJ*logJ/2. - mu*logJ +
                 mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.) * wdetJ;

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH1Diagnostic)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute Grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = Grad_u
    CeedScalar Grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        Grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          Grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (Grad_u + Grad_u^T + Grad_u^T*Grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += Grad_u[n][indj[m]]*Grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const CeedScalar Jm1 = computeJM1(Grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    diagnostic[3][i] = -lambda*logJ;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt m = 0; m < 3; m++)
        diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    diagnostic[6][i] = Jm1 + 1.;

    // Strain energy
    diagnostic[7][i] = (lambda*logJ*logJ/2. - mu*logJ +
                        mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.);

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of ELAS_FSCurrentNH1_H
