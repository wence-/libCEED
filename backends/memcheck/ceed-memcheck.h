#ifndef _ceed_memcheck_h
#define _ceed_memcheck_h

#include <ceed/ceed.h>
#include <ceed/backend.h>

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setup_done;
} CeedQFunction_Memcheck;

CEED_INTERN int CeedQFunctionCreate_Memcheck(CeedQFunction qf);

#endif // _ceed_memcheck_h
