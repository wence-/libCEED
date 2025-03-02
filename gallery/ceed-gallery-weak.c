#include <ceed/backend.h>
#include <stdbool.h>
#include <stdlib.h>

// This function provides a debug target for weak symbols
// LCOV_EXCL_START
static int CeedQFunctionRegister_Weak(const char *name) {
  CeedDebugEnv("** Weak Register: %s", name);
  return CEED_ERROR_SUCCESS;
}
// LCOV_EXCL_STOP

#define MACRO(name)                                                     \
  CEED_INTERN int name(void) __attribute__((weak));                     \
  int name(void) { return CeedQFunctionRegister_Weak(__func__); }
#include "ceed-gallery-list.h"
#undef MACRO
