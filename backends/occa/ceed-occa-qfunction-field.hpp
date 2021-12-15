#ifndef CEED_OCCA_QFUNCTIONFIELD_HEADER
#define CEED_OCCA_QFUNCTIONFIELD_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
  namespace occa {
    class QFunctionField {
     protected:
      bool _isValid;

     public:
      CeedEvalMode evalMode;
      CeedInt size;

      QFunctionField(CeedQFunctionField qfField);

      bool isValid() const;
    };
  }
}

#endif
