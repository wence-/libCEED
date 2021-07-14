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

#ifndef ELAS_FSInitialNH1_H
#define ELAS_FSInitialNH1_H

#ifndef __CUDACC__
#  include <math.h>
#endif

// typedef for single vs double precision (see issue in repo about changing ElastScalar; keep this more orthogonal)
// make typedef for 'ElastScalar' and make ElastScalar the defalut.
// add in compile time option (-qfunction_precision = single|double) to change this for reproducibility. use ElastScalar locally
#ifndef ELAST_SCALAR_DEFINED
#  ifdef ELAST_SCALAR_SINGLE
  typedef float ElastScalar;
#  else
  typedef CeedScalar ElastScalar;
#  endif
#  define ELAST_SCALAR_DEFINED
#endif

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
CEED_QFUNCTION_HELPER ElastScalar log1p_series_shifted(ElastScalar x) {
  const ElastScalar left = sqrt(2.)/2 - 1, right = sqrt(2.) - 1;
  ElastScalar sum = 0;
  if (1) { // Disable if the smaller range sqrt(2) < J < sqrt(2) is sufficient
    if (x < left) { // Replace if with while for arbitrary range (may hurt vectorization)
      sum -= log(2.) / 2;
      x = 1 + 2 * x;
    } else if (right < x) {
      sum += log(2.) / 2;
      x = (x - 1) / 2;
    }
  }
  ElastScalar y = x / (2. + x);
  const ElastScalar y2 = y*y;
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
CEED_QFUNCTION_HELPER ElastScalar computeJM1(const CeedScalar grad_u[3][3]) {
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
// Common computations between FS and dFS
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int commonFS(const CeedScalar lambda, const CeedScalar mu,
                                   const CeedScalar grad_u[3][3], ElastScalar Swork[6],
                                   ElastScalar Cinvwork[6], ElastScalar *Jm1, ElastScalar *llnj) {
  // E - Green-Lagrange strain tensor
  //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
  const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  ElastScalar E2work[6];
  for (CeedInt m = 0; m < 6; m++) {
    E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
    for (CeedInt n = 0; n < 3; n++)
      E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
  }
  // *INDENT-OFF*
  ElastScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                         {E2work[5], E2work[1], E2work[3]},
                         {E2work[4], E2work[3], E2work[2]}
                        };
  // *INDENT-ON*
  // J-1
  *Jm1 = computeJM1(grad_u);

  // C : right Cauchy-Green tensor
  // C = I + 2E
  // *INDENT-OFF*
  const ElastScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  // *INDENT-ON*

  // Compute C^(-1) : C-Inverse
  ElastScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
                     C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
                     C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
                     C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
                     C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
                     C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 6; m++)
    Cinvwork[m] = A[m] / ((*Jm1 + 1.)*(*Jm1 + 1.));

  // *INDENT-OFF*
  const ElastScalar C_inv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                 {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                 {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                };
  // *INDENT-ON*

  // Compute the Second Piola-Kirchhoff (S)
  const ElastScalar logJ = log1p_series_shifted(*Jm1);
  (*llnj) = lambda*logJ;
  for (CeedInt m = 0; m < 6; m++) {
    Swork[m] = (*llnj)*Cinvwork[m];
    for (CeedInt n = 0; n < 3; n++)
      Swork[m] += mu*C_inv[indj[m]][n]*E2[n][indk[m]];
  }

  return 0;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1F)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const ElastScalar (*ug)[3][CEED_Q_VLA] = (const ElastScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const ElastScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  ElastScalar (*dvdX)[3][CEED_Q_VLA] = (ElastScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  ElastScalar (*grad_u)[3][CEED_Q_VLA] = (ElastScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const ElastScalar E  = context->E;
  const ElastScalar nu = context->nu;
  const ElastScalar TwoMu = E / (1 + nu);
  const ElastScalar mu = TwoMu / 2;
  const ElastScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const ElastScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  C_inv  : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff (in current config)
  //  P     : 1st Piola-Kirchhoff (in referential config)
  // Formulation:
  //  F =  I3 + grad_ue
  //  J = det(F)
  //  C = F(^T)*F
  //  S = mu*I3 + (lambda*log(J)-mu)*C_inv;
  //  P = F*S

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const ElastScalar du[3][3]   = {{ug[0][0][i],
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
    const ElastScalar wdetJ      =   q_data[0][i];
    const ElastScalar dXdx[3][3] = {{q_data[1][i],
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

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        grad_u[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          grad_u[j][k][i] += dXdx[m][k] * du[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const ElastScalar F[3][3] =  {{grad_u[0][0][i] + 1,
                                  grad_u[0][1][i],
                                  grad_u[0][2][i]},
                                 {grad_u[1][0][i],
                                  grad_u[1][1][i] + 1,
                                  grad_u[1][2][i]},
                                 {grad_u[2][0][i],
                                  grad_u[2][1][i],
                                  grad_u[2][2][i] + 1}
                                };
    // *INDENT-ON*

    // Common components of finite strain calculations
    ElastScalar Swork[6], Cinvwork[6], llnj, Jm1;
    // *INDENT-OFF*
    const ElastScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*
    commonFS(lambda, mu, tempgradu, Swork, Cinvwork, &Jm1, &llnj);

    // Second Piola-Kirchhoff (S)
    // *INDENT-OFF*
    const ElastScalar S[3][3] = {{Swork[0], Swork[5], Swork[4]},
                                {Swork[5], Swork[1], Swork[3]},
                                {Swork[4], Swork[3], Swork[2]}
                               };
    // *INDENT-ON*

    // Compute the First Piola-Kirchhoff : P = F*S
    ElastScalar P[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        P[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          P[j][k] += F[j][m] * S[m][k];
      }

    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1dF)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const ElastScalar (*deltaug)[3][CEED_Q_VLA] = (const ElastScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const ElastScalar(*)[CEED_Q_VLA])in[1];
  // grad_u is used for hyperelasticity (non-linear)
  const ElastScalar (*grad_u)[3][CEED_Q_VLA] = (const ElastScalar(*)[3][CEED_Q_VLA])in[2];

  // Outputs
  ElastScalar (*deltadvdX)[3][CEED_Q_VLA] = (ElastScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const ElastScalar E  = context->E;
  const ElastScalar nu = context->nu;

  // Constants
  const ElastScalar TwoMu = E / (1 + nu);
  const ElastScalar mu = TwoMu / 2;
  const ElastScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const ElastScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of delta_u
    // *INDENT-OFF*
    const ElastScalar deltadu[3][3] = {{deltaug[0][0][i],
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
    const ElastScalar wdetJ      =      q_data[0][i];
    const ElastScalar dXdx[3][3] =    {{q_data[1][i],
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

    // Compute graddeltau
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    ElastScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m =0 ; m < 3; m++)
          graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const ElastScalar F[3][3] =      {{grad_u[0][0][i] + 1,
                                      grad_u[0][1][i],
                                      grad_u[0][2][i]},
                                     {grad_u[1][0][i],
                                      grad_u[1][1][i] + 1,
                                      grad_u[1][2][i]},
                                     {grad_u[2][0][i],
                                      grad_u[2][1][i],
                                      grad_u[2][2][i] + 1}
                                    };
    // *INDENT-ON*

    // Common components of finite strain calculations
    ElastScalar Swork[6], Cinvwork[6], llnj, Jm1;
    // *INDENT-OFF*
    const ElastScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*
    commonFS(lambda, mu, tempgradu, Swork, Cinvwork, &Jm1, &llnj);

    // deltaE - Green-Lagrange strain tensor
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    ElastScalar deltaEwork[6];
    for (CeedInt m = 0; m < 6; m++) {
      deltaEwork[m] = 0;
      for (CeedInt n = 0; n < 3; n++)
        deltaEwork[m] += (graddeltau[n][indj[m]]*F[n][indk[m]] +
                          F[n][indj[m]]*graddeltau[n][indk[m]])/2.;
    }
    // *INDENT-OFF*
    ElastScalar deltaE[3][3] = {{deltaEwork[0], deltaEwork[5], deltaEwork[4]},
                               {deltaEwork[5], deltaEwork[1], deltaEwork[3]},
                               {deltaEwork[4], deltaEwork[3], deltaEwork[2]}
                              };
    // *INDENT-ON*

    // C : right Cauchy-Green tensor
    // C^(-1) : C-Inverse
    // *INDENT-OFF*
    const ElastScalar C_inv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                   {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                   {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                  };
    // *INDENT-ON*

    // Second Piola-Kirchhoff (S)
    // *INDENT-OFF*
    const ElastScalar S[3][3] = {{Swork[0], Swork[5], Swork[4]},
                                {Swork[5], Swork[1], Swork[3]},
                                {Swork[4], Swork[3], Swork[2]}
                               };
    // *INDENT-ON*

    // deltaS = dSdE:deltaE
    //      = lambda(C_inv:deltaE)C_inv + 2(mu-lambda*log(J))C_inv*deltaE*C_inv
    // -- C_inv:deltaE
    ElastScalar Cinv_contract_E = 0;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++)
        Cinv_contract_E += C_inv[j][k]*deltaE[j][k];
    // -- deltaE*C_inv
    ElastScalar deltaECinv[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaECinv[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaECinv[j][k] += deltaE[j][m]*C_inv[m][k];
      }
    // -- intermediate deltaS = C_inv*deltaE*C_inv
    ElastScalar deltaS[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaS[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaS[j][k] += C_inv[j][m]*deltaECinv[m][k];
      }
    // -- deltaS = lambda(C_inv:deltaE)C_inv - 2(lambda*log(J)-mu)*(intermediate)
    const ElastScalar llnj_m = llnj - mu;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++)
        deltaS[j][k] = lambda*Cinv_contract_E*C_inv[j][k] -
                       2.*llnj_m*deltaS[j][k];

    // deltaP = dPdF:deltaF = deltaF*S + F*deltaS
    ElastScalar deltaP[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaP[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaP[j][k] += graddeltau[j][m]*S[m][k] + F[j][m]*deltaS[m][k];
      }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * deltaP[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1Energy)(void *ctx, CeedInt Q,
                                       const CeedScalar *const *in,
                                       CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const ElastScalar (*ug)[3][CEED_Q_VLA] = (const ElastScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const ElastScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  ElastScalar (*energy) = (ElastScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const ElastScalar E  = context->E;
  const ElastScalar nu = context->nu;
  const ElastScalar TwoMu = E / (1 + nu);
  const ElastScalar mu = TwoMu / 2;
  const ElastScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const ElastScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const ElastScalar du[3][3]   = {{ug[0][0][i],
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
    const ElastScalar wdetJ      =   q_data[0][i];
    const ElastScalar dXdx[3][3] = {{q_data[1][i],
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

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    ElastScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    ElastScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    ElastScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const ElastScalar Jm1 = computeJM1(grad_u);
    const ElastScalar logJ = log1p_series_shifted(Jm1);

    // Strain energy Phi(E) for compressible Neo-Hookean
    energy[i] = (lambda*logJ*logJ/2. - mu*logJ +
                 mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.) * wdetJ;

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialNH1Diagnostic)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const ElastScalar (*u)[CEED_Q_VLA] = (const ElastScalar(*)[CEED_Q_VLA])in[0],
                   (*ug)[3][CEED_Q_VLA] = (const ElastScalar(*)[3][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const ElastScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  ElastScalar (*diagnostic)[CEED_Q_VLA] = (ElastScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const ElastScalar E  = context->E;
  const ElastScalar nu = context->nu;
  const ElastScalar TwoMu = E / (1 + nu);
  const ElastScalar mu = TwoMu / 2;
  const ElastScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const ElastScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const ElastScalar du[3][3]   = {{ug[0][0][i],
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
    const ElastScalar dXdx[3][3] = {{q_data[1][i],
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

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    ElastScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    ElastScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    ElastScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const ElastScalar Jm1 = computeJM1(grad_u);
    const ElastScalar logJ = log1p_series_shifted(Jm1);
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

#endif // End of ELAS_FSInitialNH1_H
