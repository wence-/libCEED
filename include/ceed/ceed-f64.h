/// @file
/// Public header for definitions related to using FP64 floating point (double
/// precision) for CeedScalar. This is the default header included in ceed/ceed.h. 
#ifndef _ceed_f64_h
#define _ceed_f64_h

/// Set base scalar type to FP64.  (See CeedScalarType enum in ceed/ceed.h
/// for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP64
typedef double CeedScalar;

/// Machine epsilon
#define CEED_EPSILON 1e-16

#endif
