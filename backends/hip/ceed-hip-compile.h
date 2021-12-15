#ifndef _ceed_hip_compile_h
#define _ceed_hip_compile_h

#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

CEED_INTERN int CeedCompileHip(Ceed ceed, const char *source,
                               hipModule_t *module,
                               const CeedInt numopts, ...);

CEED_INTERN int CeedGetKernelHip(Ceed ceed, hipModule_t module,
                                 const char *name,
                                 hipFunction_t *kernel);

CEED_INTERN int CeedRunKernelHip(Ceed ceed, hipFunction_t kernel,
                                 const int gridSize,
                                 const int blockSize, void **args);

CEED_INTERN int CeedRunKernelDimHip(Ceed ceed, hipFunction_t kernel,
                                    const int gridSize,
                                    const int blockSizeX, const int blockSizeY,
                                    const int blockSizeZ, void **args);

CEED_INTERN int CeedRunKernelDimSharedHip(Ceed ceed, hipFunction_t kernel,
    const int gridSize, const int blockSizeX,
    const int blockSizeY, const int blockSizeZ,
    const int sharedMemSize, void **args);

#endif // _ceed_hip_compile_h
