/**
  @brief Ceed QFunction for building the geometric data for the 2D mass matrix
**/

#ifndef mass2dbuild_h
#define mass2dbuild_h

CEED_QFUNCTION(Mass2DBuild)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *w = in[1];
  // out[0] is quadrature data, size (Q)
  CeedScalar *q_data = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    q_data[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * w[i];
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // mass2dbuild_h
