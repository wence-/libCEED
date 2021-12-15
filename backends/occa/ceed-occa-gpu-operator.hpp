#ifndef CEED_OCCA_GPU_OPERATOR_HEADER
#define CEED_OCCA_GPU_OPERATOR_HEADER

#include <vector>

#include "ceed-occa-operator.hpp"

namespace ceed {
  namespace occa {
    class GpuOperator : public Operator {
     public:
      GpuOperator();

      ~GpuOperator();

      ::occa::kernel buildApplyAddKernel();

      void applyAdd(Vector *in, Vector *out);
    };
  }
}

#endif
