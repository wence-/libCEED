#ifndef CEED_OCCA_KERNELS_ELEMRESTRICTION_HEADER
#define CEED_OCCA_KERNELS_ELEMRESTRICTION_HEADER

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - COMPONENT_COUNT            : CeedInt
// - ELEMENT_SIZE               : CeedInt
// - NODE_COUNT                 : CeedInt
// - TILE_SIZE                  : int
// - USES_INDICES               : bool
// - STRIDE_TYPE                : ceed::occa::StrideType
// - NODE_STRIDE                : Optional[CeedInt]
// - COMPONENT_STRIDE           : Optional[CeedInt]
// - ELEMENT_STRIDE             : Optional[CeedInt]
// - UNSTRIDED_COMPONENT_STRIDE : Optional[CeedInt]

extern const char *occa_elem_restriction_source;

#endif
