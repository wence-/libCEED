CEED_QFUNCTION(setup_mass)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  const CeedScalar *J = in[0], *weight = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }
  return 0;
}

CEED_QFUNCTION(apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // in[0] is u, size (Q)
  // in[1] is mass quadrature data, size (Q)
  const CeedScalar *u = in[0], *qd_mass = in[1];

  // out[0] is output to multiply against v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // Mass
    v[i] = qd_mass[i]*u[i];
  }

  return 0;
}
