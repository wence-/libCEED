/// @file
/// libCEED QFunctions for BP examples using PETSc

#ifndef common_h
#define common_h

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Error)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *target = in[1];
  CeedScalar *error = out[0];
  for (CeedInt i=0; i<Q; i++) {
    error[i] = u[i] - target[i];
  }
  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Error3)(void *ctx, CeedInt Q,
                       const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *target = in[1];
  CeedScalar *error = out[0];
  for (CeedInt i=0; i<3*Q; i++) {
    error[i] = u[i] - target[i];
  }
  return 0;
}
// -----------------------------------------------------------------------------

#endif // common_h
