/**
  @brief Ceed QFunction for applying the mass matrix
**/

#ifndef massapply_h
#define massapply_h

CEED_QFUNCTION(MassApply)(void *ctx, const CeedInt Q,
                          const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is u, size (Q)
  // in[1] is quadrature data, size (Q)
  const CeedScalar *u = in[0], *q_data = in[1];
  // out[0] is v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    v[i] = u[i] * q_data[i];
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // massapply_h
