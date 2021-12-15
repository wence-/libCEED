#ifndef CEED_OCCA_CEEDOBJECT_HEADER
#define CEED_OCCA_CEEDOBJECT_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
  namespace occa {
    class CeedObject {
     private:
      ::occa::device _device;

     public:
      Ceed ceed;

      CeedObject(Ceed ceed_ = NULL);

      ::occa::device getDevice();

      bool usingCpuDevice() const;
      bool usingGpuDevice() const;

      int ceedError(const std::string &message) const;
      static int staticCeedError(const std::string &message);
    };

    namespace SyncState {
      static const int none   = 0;
      static const int host   = (1 << 0);
      static const int device = (1 << 1);
      static const int all    = host | device;
    }
  }
}

#endif
