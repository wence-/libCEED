#include <cuda.h>    // for CUDA_VERSION
#include "../common/weight.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int Q>
static magma_int_t 
magma_weight_3d_kernel_driver(
    const T *dqweight1d, T *dV, magma_int_t v_stride, 
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;

    magma_int_t nthreads = (Q*Q); 
    magma_int_t ntcol = (maxthreads < nthreads) ? 1 : (maxthreads / nthreads);
    magma_int_t shmem  = 0;
    shmem += sizeof(T) * Q;  // for dqweight1d 

    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(magma_weight_3d_kernel<T, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    if ( (nthreads*ntcol) > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        magma_int_t nblocks = (nelem + ntcol-1) / ntcol;
        dim3 threads(nthreads, ntcol, 1);
        dim3 grid(nblocks, 1, 1);
        magma_weight_3d_kernel<T, Q><<<grid, threads, shmem, magma_queue_get_cuda_stream(queue)>>>
        (dqweight1d, dV, v_stride, nelem);
        return (cudaPeekAtLastError() == cudaSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_weight_3d_q(
        magma_int_t Q, const CeedScalar *dqweight1d, 
        CeedScalar *dV, magma_int_t v_stride, 
        magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 1>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  2: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 2>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  3: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 3>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  4: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 4>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  5: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 5>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  6: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 6>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  7: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 7>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  8: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 8>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case  9: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar, 9>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        case 10: 
          launch_failed = magma_weight_3d_kernel_driver<CeedScalar,10>
          (dqweight1d, dV, v_stride, nelem, maxthreads, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_weight_3d( 
    magma_int_t Q, const CeedScalar *dqweight1d, 
    CeedScalar *dV, magma_int_t v_stride, 
    magma_int_t nelem, magma_int_t maxthreads, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_weight_3d_q(Q, dqweight1d, dV, v_stride, nelem, maxthreads, queue);
    return launch_failed;
}
