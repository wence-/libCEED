#ifndef _ceed_xsmm_h
#define _ceed_xsmm_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/hash.h>
#include <libxsmm.h>

// Instantiate khash structs and methods
CeedHashIJKLMInit(f32, libxsmm_smmfunction)
CeedHashIJKLMInit(f64, libxsmm_dmmfunction)

typedef struct {
  bool is_tensor;
  CeedInt P, Q, dim;
  khash_t(f32) *lookup_f32;
  khash_t(f64) *lookup_f64;
} CeedTensorContract_Xsmm;

CEED_INTERN int CeedTensorContractCreate_f32_Xsmm(CeedBasis basis,
    CeedTensorContract contract);

CEED_INTERN int CeedTensorContractCreate_f64_Xsmm(CeedBasis basis,
    CeedTensorContract contract);

#endif // _ceed_xsmm_h
