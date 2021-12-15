#ifndef CEED_OCCA_SIMPLEXBASIS_HEADER
#define CEED_OCCA_SIMPLEXBASIS_HEADER

#include "ceed-occa-basis.hpp"

namespace ceed {
  namespace occa {
    class SimplexBasis : public Basis {
     public:
      ::occa::memory interp;
      ::occa::memory grad;
      ::occa::memory qWeight;
      ::occa::kernelBuilder interpKernelBuilder;
      ::occa::kernelBuilder gradKernelBuilder;
      ::occa::kernelBuilder weightKernelBuilder;

      SimplexBasis(CeedBasis basis,
                   CeedInt dim,
                   CeedInt P_,
                   CeedInt Q_,
                   const CeedScalar *interp_,
                   const CeedScalar *grad_,
                   const CeedScalar *qWeight_);

      ~SimplexBasis();

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
                                        const bool transpose);

      int apply(const CeedInt elementCount,
                CeedTransposeMode tmode,
                CeedEvalMode emode,
                Vector *u,
                Vector *v);

      //---[ Ceed Callbacks ]-----------
      static int ceedCreate(CeedElemTopology topology,
                            CeedInt dim,
                            CeedInt ndof,
                            CeedInt nquad,
                            const CeedScalar *interp,
                            const CeedScalar *grad,
                            const CeedScalar *qref,
                            const CeedScalar *qWeight,
                            CeedBasis basis);
    };
  }
}

#endif
