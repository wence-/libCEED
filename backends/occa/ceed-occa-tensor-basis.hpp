#ifndef CEED_OCCA_TENSORBASIS_HEADER
#define CEED_OCCA_TENSORBASIS_HEADER

#include "ceed-occa-basis.hpp"

namespace ceed {
  namespace occa {
    class TensorBasis : public Basis {
     public:
      CeedInt P1D;
      CeedInt Q1D;
      ::occa::memory interp1D;
      ::occa::memory grad1D;
      ::occa::memory qWeight1D;
      ::occa::kernelBuilder interpKernelBuilder;
      ::occa::kernelBuilder gradKernelBuilder;
      ::occa::kernelBuilder weightKernelBuilder;

      TensorBasis(CeedBasis basis,
                  CeedInt dim_,
                  CeedInt P1D_,
                  CeedInt Q1D_,
                  const CeedScalar *interp1D_,
                  const CeedScalar *grad1D_,
                  const CeedScalar *qWeight1D_);

      ~TensorBasis();

      bool isTensorBasis() const;

      const char* getFunctionSource() const;

      void setupKernelBuilders();

      int applyInterp(const CeedInt elementCount,
                      const bool transpose,
                      Vector &U,
                      Vector &V);

      ::occa::kernel getCpuInterpKernel(const bool transpose);
      ::occa::kernel getGpuInterpKernel(const bool transpose);

      int applyGrad(const CeedInt elementCount,
                    const bool transpose,
                    Vector &U,
                    Vector &V);

      ::occa::kernel getCpuGradKernel(const bool transpose);
      ::occa::kernel getGpuGradKernel(const bool transpose);

      int applyWeight(const CeedInt elementCount,
                      Vector &W);

      ::occa::kernel getCpuWeightKernel();
      ::occa::kernel getGpuWeightKernel();

      ::occa::kernel buildCpuEvalKernel(::occa::kernelBuilder &kernelBuilder,
                                        const bool transpose);

      ::occa::kernel buildGpuEvalKernel(::occa::kernelBuilder &kernelBuilder,
                                        const bool transpose,
                                        const int elementsPerBlock);

      int apply(const CeedInt elementCount,
                CeedTransposeMode tmode,
                CeedEvalMode emode,
                Vector *U,
                Vector *V);

      //---[ Ceed Callbacks ]-----------
      static int ceedCreate(CeedInt dim,
                            CeedInt P1D,
                            CeedInt Q1D,
                            const CeedScalar *interp1D,
                            const CeedScalar *grad1D,
                            const CeedScalar *qref1D,
                            const CeedScalar *qWeight1D,
                            CeedBasis basis);
    };
  }
}

#endif
