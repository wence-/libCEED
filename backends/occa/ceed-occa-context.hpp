#ifndef CEED_OCCA_CONTEXT_HEADER
#define CEED_OCCA_CONTEXT_HEADER

#include "ceed-occa-types.hpp"

namespace ceed {
  namespace occa {
    class Context {
     private:
      bool _usingCpuDevice;
      bool _usingGpuDevice;

     public:
      ::occa::device device;

      Context(::occa::device device_);

      static Context* from(Ceed ceed);

      bool usingCpuDevice() const;
      bool usingGpuDevice() const;
    };
  }
}

#endif
