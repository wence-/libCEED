#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-identity.h"

/**
  @brief Set fields identity QFunction that copies inputs directly into outputs
**/
static int CeedQFunctionInit_Identity(Ceed ceed, const char *requested,
                                      CeedQFunction qf) {
  // Check QFunction name
  const char *name = "Identity";
  if (strcmp(name, requested))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "QFunction '%s' does not match requested name: %s",
                     name, requested);
  // LCOV_EXCL_STOP

  // QFunction fields 'input' and 'output' with requested emodes added
  //   by the library rather than being added here

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register identity QFunction that copies inputs directly into outputs
**/
CEED_INTERN int CeedQFunctionRegister_Identity(void) {
  return CeedQFunctionRegister("Identity", Identity_loc, 1, Identity,
                               CeedQFunctionInit_Identity);
}
