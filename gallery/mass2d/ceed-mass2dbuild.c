#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-mass2dbuild.h"

/**
  @brief Set fields for Ceed QFunction building the geometric data for the 2D
           mass matrix
**/
static int CeedQFunctionInit_Mass2DBuild(Ceed ceed, const char *requested,
    CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "Mass2DBuild";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  const CeedInt dim = 2;
  ierr = CeedQFunctionAddInput(qf, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "weights", 1, CEED_EVAL_WEIGHT);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for building the geometric data for the 2D mass
           matrix
**/
CEED_INTERN int CeedQFunctionRegister_Mass2DBuild(void) {
  return CeedQFunctionRegister("Mass2DBuild", Mass2DBuild_loc, 1, Mass2DBuild,
                               CeedQFunctionInit_Mass2DBuild);
}
