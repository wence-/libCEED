/**
  @brief Ceed QFunction for building the geometric data for the 1D Poisson operator
**/

#ifndef poisson1dbuild_h
#define poisson1dbuild_h

CEED_QFUNCTION(Poisson1DBuild)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians, size (Q)
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *w = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *q_data = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    q_data[i] = w[i] / J[i];
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson1dbuild_h
