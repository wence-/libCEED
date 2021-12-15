#ifndef CEED_OCCA_KERNELS_SIMPLEXBASIS_HEADER
#define CEED_OCCA_KERNELS_SIMPLEXBASIS_HEADER

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - DIM                  : CeedInt
// - Q                    : CeedInt
// - P                    : CeedInt
// - MAX_PQ               : CeedInt
// - BASIS_COMPONENT_COUNT: CeedInt
// - ELEMENTS_PER_BLOCK   : CeedInt
// - TRANSPOSE            : bool

extern const char *occa_simplex_basis_cpu_function_source;
extern const char *occa_simplex_basis_cpu_kernel_source;

extern const char *occa_simplex_basis_gpu_source;

#endif
