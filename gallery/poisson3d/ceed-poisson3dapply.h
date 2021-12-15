/**
  @brief Ceed QFunction for applying the geometric data for the 3D Poisson
           operator
**/

#ifndef poisson3dapply_h
#define poisson3dapply_h

CEED_QFUNCTION(Poisson3DApply)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // in[0] is gradient u, shape [3, nc=1, Q]
  // in[1] is quadrature data, size (6*Q)
  const CeedScalar *ug = in[0], *q_data = in[1];

  // out[0] is output to multiply against gradient v, shape [3, nc=1, Q]
  CeedScalar *vg = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[3]        =  {ug[i+Q*0],
                                      ug[i+Q*1],
                                      ug[i+Q*2]
                                     };

    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 5 4
    // 5 1 3
    // 4 3 2
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[3][3] = {{q_data[i+0*Q],
                                         q_data[i+5*Q],
                                         q_data[i+4*Q]},
                                        {q_data[i+5*Q],
                                         q_data[i+1*Q],
                                         q_data[i+3*Q]},
                                        {q_data[i+4*Q],
                                         q_data[i+3*Q],
                                         q_data[i+2*Q]}
                                       };
    // *INDENT-ON*

    // Apply Poisson Operator
    // j = direction of vg
    for (int j=0; j<3; j++)
      vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                   du[1] * dXdxdXdxT[1][j] +
                   du[2] * dXdxdXdxT[2][j]);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson3dapply_h
