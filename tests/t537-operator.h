CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    for (CeedInt c=0; c<2; c++)
      v[i+Q*c] = rho[i] * u[i+Q*(c ? 0 : 1)];
  }
  return 0;
}
