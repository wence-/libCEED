#ifndef _helper_h
#define _helper_h

CEED_QFUNCTION_HELPER CeedScalar times_two(CeedScalar x) {
  return 2 * x;
}

CEED_QFUNCTION_HELPER CeedScalar times_three(CeedScalar x) {
  return 3 * x;
}

#endif
