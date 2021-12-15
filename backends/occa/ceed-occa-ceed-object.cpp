#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-context.hpp"

namespace ceed {
  namespace occa {
    CeedObject::CeedObject(Ceed ceed_) :
        ceed(ceed_) {}

    ::occa::device CeedObject::getDevice() {
      if (!_device.isInitialized()) {
        _device = Context::from(ceed)->device;
      }
      return _device;
    }

    bool CeedObject::usingCpuDevice() const {
      return Context::from(ceed)->usingCpuDevice();
    }

    bool CeedObject::usingGpuDevice() const {
      return Context::from(ceed)->usingGpuDevice();
    }

    int CeedObject::ceedError(const std::string &message) const {
      return CeedError(ceed, CEED_ERROR_BACKEND, message.c_str());
    }

    int CeedObject::staticCeedError(const std::string &message) {
      return CeedError(NULL, CEED_ERROR_BACKEND, message.c_str());
    }
  }
}
