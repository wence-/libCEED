/**
  @brief  Identity QFunction that copies inputs directly into outputs
**/

#ifndef identity_h
#define identity_h

CEED_QFUNCTION(Identity)(void *ctx, const CeedInt Q,
                         const CeedScalar *const *in,
                         CeedScalar *const *out) {
  // Ctx holds field size
  const CeedInt size = *(CeedInt *)ctx;

  // in[0] is input, size (Q*size)
  const CeedScalar *input = in[0];
  // out[0] is output, size (Q*size)
  CeedScalar *output = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q*size; i++) {
    output[i] = input[i];
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // identity_h
