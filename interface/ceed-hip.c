#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/hip.h>
#include <ceed-impl.h>

/**
  @brief Set HIP function pointer to evaluate action at quadrature points

  @param qf  CeedQFunction to set device pointer
  @param f   Device function pointer to evaluate action at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetHIPUserFunction(CeedQFunction qf, hipFunction_t f) {
  int ierr;
  if (!qf->SetHIPUserFunction) {
    Ceed ceed;
    ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
    CeedDebug(ceed,
              "Backend does not support hipFunction_t pointers for QFunctions.");
  } else {
    ierr = qf->SetHIPUserFunction(qf, f); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}
