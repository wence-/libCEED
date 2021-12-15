#ifndef _ceed_avx_h
#define _ceed_avx_h

#include <ceed/ceed.h>
#include <ceed/backend.h>

CEED_INTERN int CeedTensorContractCreate_f32_Avx(CeedBasis basis,
    CeedTensorContract contract);
CEED_INTERN int CeedTensorContractCreate_f64_Avx(CeedBasis basis,
    CeedTensorContract contract);

#endif // _ceed_avx_h
