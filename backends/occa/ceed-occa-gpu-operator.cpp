#include "ceed-occa-gpu-operator.hpp"
#include "ceed-occa-qfunction.hpp"

namespace ceed {
  namespace occa {
    GpuOperator::GpuOperator() {}

    GpuOperator::~GpuOperator() {}

    ::occa::kernel GpuOperator::buildApplyAddKernel() {
      return ::occa::kernel();
    }

    void GpuOperator::applyAdd(Vector *in, Vector *out) {
      // TODO: Implement
    }
  }
}
