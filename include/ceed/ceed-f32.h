/// @file
/// Public header for definitions related to using FP32 floating point (single
/// precision) for CeedScalar.  Include this header in ceed/ceed.h to use 
/// float instead of double. 
#ifndef _ceed_f32_h
#define _ceed_f32_h

/// Set base scalar type to FP32.  (See CeedScalarType enum in ceed/ceed.h
/// for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP32
typedef float CeedScalar;

/// Machine epsilon
#define CEED_EPSILON 6e-08

#endif
