#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-poisson3dbuild.h"

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 3D
           Poisson operator
**/
static int CeedQFunctionInit_Poisson3DBuild(Ceed ceed, const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Poisson3DBuild";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt dim = 3;
  ierr = CeedQFunctionAddInput(qf, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
  CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 3D
           Poisson operator
**/
CEED_INTERN int CeedQFunctionRegister_Poisson3DBuild(void) {
  return CeedQFunctionRegister("Poisson3DBuild", Poisson3DBuild_loc, 1,
                               Poisson3DBuild, CeedQFunctionInit_Poisson3DBuild);
}
