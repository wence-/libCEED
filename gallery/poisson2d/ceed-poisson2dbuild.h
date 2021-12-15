/**
  @brief Ceed QFunction for building the geometric data for the 2D Poisson operator
**/

#ifndef poisson2dbuild_h
#define poisson2dbuild_h

CEED_QFUNCTION(Poisson2DBuild)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *w = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *q_data = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Qdata stored in Voigt convention
    // J: 0 2   q_data: 0 2   adj(J):  J22 -J12
    //    1 3           2 1           -J21  J11
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J12 = J[i+Q*2];
    const CeedScalar J22 = J[i+Q*3];
    const CeedScalar qw = w[i] / (J11*J22 - J21*J12);
    q_data[i+Q*0] =   qw * (J12*J12 + J22*J22);
    q_data[i+Q*1] =   qw * (J11*J11 + J21*J21);
    q_data[i+Q*2] = - qw * (J11*J12 + J21*J22);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson2dbuild_h
