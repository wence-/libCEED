/**
  @brief Ceed QFunction for applying the 2D Poisson operator
**/

#ifndef poisson2dapply_h
#define poisson2dapply_h

CEED_QFUNCTION(Poisson2DApply)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar *ug = in[0], *q_data = in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *vg = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[2]        =  {ug[i+Q*0],
                                      ug[i+Q*1]
                                     };

    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[2][2] = {{q_data[i+0*Q],
                                         q_data[i+2*Q]},
                                        {q_data[i+2*Q],
                                         q_data[i+1*Q]}
                                       };
    // *INDENT-ON*

    // Apply Poisson operator
    // j = direction of vg
    for (int j=0; j<2; j++)
      vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                   du[1] * dXdxdXdxT[1][j]);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson2dapply_h
