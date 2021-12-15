#ifndef CEED_MAGMA_ATOMICS_HIP_H
#define CEED_MAGMA_ATOMICS_HIP_H

#include "magma_internal.h"
/******************************************************************************/
// Atomic adds 
/******************************************************************************/
__device__ static __inline__ float 
magmablas_satomic_add(float* address, float val)
{
    return atomicAdd(address, val);
}

/******************************************************************************/
__device__ static __inline__ double 
magmablas_datomic_add(double* address, double val)
{
   return atomicAdd(address, val);
}

#endif // CEED_MAGMA_ATOMICS_HIP_H
