#include <ceed/ceed.h>
#include <cuda.h>

const int sizeMax = 16;
__constant__ CeedScalar c_B[sizeMax*sizeMax];
__constant__ CeedScalar c_G[sizeMax*sizeMax];

//------------------------------------------------------------------------------
// Interp device initalization
//------------------------------------------------------------------------------
extern "C" int CeedCudaInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                                  CeedScalar **c_B_ptr) {
  const int Bsize = P1d*Q1d*sizeof(CeedScalar);
  cudaMemcpyToSymbol(c_B, d_B, Bsize, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);

  return 0;
}

//------------------------------------------------------------------------------
// Grad device initalization
//------------------------------------------------------------------------------
extern "C" int CeedCudaInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G,
    CeedInt P1d, CeedInt Q1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
  const int Bsize = P1d*Q1d*sizeof(CeedScalar);
  cudaMemcpyToSymbol(c_B, d_B, Bsize, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
  cudaMemcpyToSymbol(c_G, d_G, Bsize, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_G_ptr, c_G);

  return 0;
}
//------------------------------------------------------------------------------
