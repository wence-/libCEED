/**
  @brief Ceed QFunction for applying the 1D Poisson operator
**/

#ifndef poisson1dapply_h
#define poisson1dapply_h

CEED_QFUNCTION(Poisson1DApply)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // in[0] is gradient u, size (Q)
  // in[1] is quadrature data, size (Q)
  const CeedScalar *du = in[0], *q_data = in[1];

  // out[0] is output to multiply against gradient v, size (Q)
  CeedScalar *dv = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    dv[i] = du[i] * q_data[i];
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson1dapply_h
