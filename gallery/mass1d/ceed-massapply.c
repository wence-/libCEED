#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-massapply.h"

/**
  @brief Set fields for Ceed QFunction for applying the mass matrix
**/
static int CeedQFunctionInit_MassApply(Ceed ceed, const char *requested,
                                       CeedQFunction qf) {
  int ierr;

  // Check QFunction name
  const char *name = "MassApply";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // Add QFunction fields
  ierr = CeedQFunctionAddInput(qf, "u", 1, CEED_EVAL_INTERP); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf, "qdata", 1, CEED_EVAL_NONE); CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf, "v", 1, CEED_EVAL_INTERP); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register Ceed QFunction for applying the mass matrix
**/
CEED_INTERN int CeedQFunctionRegister_MassApply(void) {
  return CeedQFunctionRegister("MassApply", MassApply_loc, 1, MassApply,
                               CeedQFunctionInit_MassApply);
}
