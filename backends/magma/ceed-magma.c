// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "ceed-magma.h"

//------------------------------------------------------------------------------
// RTC
//------------------------------------------------------------------------------
int magma_rtc_cuda(Ceed ceed, const char *source, CUmodule *module,
              const CeedInt numopts, ...)
{
  int ierr;
  nvrtcProgram prog;
  CeedChk_Nvrtc(ceed, nvrtcCreateProgram(&prog, source, NULL, 0, NULL, NULL));

  // Get kernel specific options, such as kernel constants
  const int optslen = 32;
  const int optsextra = 4;
  const char *opts[numopts + optsextra];
  char buf[numopts][optslen];
  if (numopts > 0) {
    va_list args;
    va_start(args, numopts);
    char *name;
    int val;
    for (int i = 0; i < numopts; i++) {
      name = va_arg(args, char *);
      val = va_arg(args, int);
      snprintf(&buf[i][0], optslen,"-D%s=%d", name, val);
      opts[i] = &buf[i][0];
    }
    va_end(args);
  }

  // Standard backend options
  opts[numopts]     = "-DCeedScalar=double";
  opts[numopts + 1] = "-DCeedInt=int";
  opts[numopts + 2] = "-default-device";
  struct cudaDeviceProp prop;
  Ceed_Magma *magma_data;
  ierr = CeedGetData(ceed, (void *)&magma_data); CeedChk(ierr);
  ierr = cudaGetDeviceProperties(&prop, magma_data->device);
  CeedChk_Cu(ceed, ierr);
  char buff[optslen];
  snprintf(buff, optslen,"-arch=compute_%d%d", prop.major, prop.minor);
  opts[numopts + 3] = buff;

  // Compile kernel
  nvrtcResult result = nvrtcCompileProgram(prog, numopts + optsextra, opts);
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLogSize(prog, &logsize));
    char *log;
    ierr = CeedMalloc(logsize, &log); CeedChk(ierr);
    CeedChk_Nvrtc(ceed, nvrtcGetProgramLog(prog, log));
    return CeedError(ceed, (int)result, "%s\n%s", nvrtcGetErrorString(result), log);
  }

  size_t ptxsize;
  CeedChk_Nvrtc(ceed, nvrtcGetPTXSize(prog, &ptxsize));
  char *ptx;
  ierr = CeedMalloc(ptxsize, &ptx); CeedChk(ierr);
  CeedChk_Nvrtc(ceed, nvrtcGetPTX(prog, ptx));
  CeedChk_Nvrtc(ceed, nvrtcDestroyProgram(&prog));

  CeedChk_Cu(ceed, cuModuleLoadData(module, ptx));
  ierr = CeedFree(&ptx); CeedChk(ierr);
  return 0;
}

static int CeedDestroy_Magma(Ceed ceed) {
  int ierr;
  Ceed_Magma *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);
  magma_queue_destroy( data->queue );
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

static int CeedInit_Magma(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/gpu/magma"))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  CeedInit("/gpu/cuda/ref", &ceedref);
  ierr = CeedSetDelegate(ceed, ceedref); CeedChk(ierr);

  ierr = magma_init();
  if (ierr)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "error in magma_init(): %d\n", ierr);
  // LCOV_EXCL_STOP

  Ceed_Magma *data;
  ierr = CeedCalloc(sizeof(Ceed_Magma), &data); CeedChk(ierr);
  ierr = CeedSetData(ceed, (void *)&data); CeedChk(ierr);

  // kernel selection
  data->basis_kernel_mode = MAGMA_KERNEL_DIM_SPECIFIC;

  // kernel max threads per thread-block
  data->maxthreads[0] = 128;  // for 1D kernels
  data->maxthreads[1] = 128;  // for 2D kernels
  data->maxthreads[2] =  64;  // for 3D kernels

  // create a queue that uses the null stream
  magma_getdevice( &(data->device) );
  magma_queue_create_from_cuda(data->device, NULL, NULL, NULL, &(data->queue));

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate",
                                CeedCompositeOperatorCreate_Magma); CeedChk(ierr);


  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/magma", CeedInit_Magma, 20);
}
