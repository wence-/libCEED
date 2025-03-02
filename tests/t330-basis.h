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

// Hdiv basis for quadrilateral linear BDMelement in 2D
// Local numbering is as follow (each edge has 2 vector dof)
//     b4     b5
//    2---------3
//  b7|         |b3
//    |         |
//  b6|         |b2
//    0---------1
//     b0     b1
// Bx[0-->7] = b0_x-->b7_x, By[0-->7] = b0_y-->b7_y
// To see how the nodal basis is constructed visit:
// https://github.com/rezgarshakeri/H-div-Tests
int NodalHdivBasisQuad(CeedScalar *X, CeedScalar *Bx, CeedScalar *By) {
  CeedScalar x_hat = X[0];
  CeedScalar y_hat = X[1];
  Bx[0] = -0.125 + 0.125*x_hat*x_hat;
  By[0] = -0.25 + 0.25*x_hat + 0.25*y_hat + -0.25*x_hat*y_hat;
  Bx[1] = 0.125 + -0.125*x_hat*x_hat;
  By[1] = -0.25 + -0.25*x_hat + 0.25*y_hat + 0.25*x_hat*y_hat;
  Bx[2] = 0.25 + 0.25*x_hat + -0.25*y_hat + -0.25*x_hat*y_hat;
  By[2] = -0.125 + 0.125*y_hat*y_hat;
  Bx[3] = 0.25 + 0.25*x_hat + 0.25*y_hat + 0.25*x_hat*y_hat;
  By[3] = 0.125 + -0.125*y_hat*y_hat;
  Bx[4] = -0.125 + 0.125*x_hat*x_hat;
  By[4] = 0.25 + -0.25*x_hat + 0.25*y_hat + -0.25*x_hat*y_hat;
  Bx[5] = 0.125 + -0.125*x_hat*x_hat;
  By[5] = 0.25 + 0.25*x_hat + 0.25*y_hat + 0.25*x_hat*y_hat;
  Bx[6] = -0.25 + 0.25*x_hat + 0.25*y_hat + -0.25*x_hat*y_hat;
  By[6] = -0.125 + 0.125*y_hat*y_hat;
  Bx[7] = -0.25 + 0.25*x_hat + -0.25*y_hat + 0.25*x_hat*y_hat;
  By[7] = 0.125 + -0.125*y_hat*y_hat;
  return 0;
}
static void HdivBasisQuad(CeedInt Q, CeedScalar *q_ref, CeedScalar *q_weights,
                          CeedScalar *interp, CeedScalar *div, CeedQuadMode quad_mode) {

  // Get 1D quadrature on [-1,1]
  CeedScalar q_ref_1d[Q], q_weight_1d[Q];
  switch (quad_mode) {
  case CEED_GAUSS:
    CeedGaussQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  // LCOV_EXCL_START
  case CEED_GAUSS_LOBATTO:
    CeedLobattoQuadrature(Q, q_ref_1d, q_weight_1d);
    break;
  }
  // LCOV_EXCL_STOP

  // Divergence operator; Divergence of nodal basis for ref element
  CeedScalar D[8] = {0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25};
  // Loop over quadrature points
  CeedScalar Bx[8], By[8];
  CeedScalar X[2];

  for (CeedInt i=0; i<Q; i++) {
    for (CeedInt j=0; j<Q; j++) {
      CeedInt k1 = Q*i+j;
      q_ref[k1] = q_ref_1d[j];
      q_ref[k1 + Q*Q] = q_ref_1d[i];
      q_weights[k1] = q_weight_1d[j]*q_weight_1d[i];
      X[0] = q_ref_1d[j];
      X[1] = q_ref_1d[i];
      NodalHdivBasisQuad(X, Bx, By);
      for (CeedInt k=0; k<8; k++) {
        interp[k1*8+k] = Bx[k];
        interp[k1*8+k+8*Q*Q] = By[k];
        div[k1*8+k] = D[k];
      }
    }
  }
}