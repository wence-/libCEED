#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-cuda-shared.h"
#include "../cuda/ceed-cuda.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_shared(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 9; // number of characters in resource
  if (strncmp(resource, "/gpu/cuda/shared", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cuda backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  Ceed_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  ierr = CeedSetData(ceed, data); CeedChk(ierr);
  ierr = CeedCudaInit(ceed, resource, nrc); CeedChk(ierr);

  Ceed ceedref;
  CeedInit("/gpu/cuda/ref", &ceedref);
  ierr = CeedSetDelegate(ceed, ceedref); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Cuda_shared);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Cuda); CeedChk(ierr);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Shared(void) {
  return CeedRegister("/gpu/cuda/shared", CeedInit_Cuda_shared, 25);
}
//------------------------------------------------------------------------------
