#ifndef CEED_OCCA_OPERATORARGS_HEADER
#define CEED_OCCA_OPERATORARGS_HEADER

#include <vector>

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-qfunction-args.hpp"
#include "ceed-occa-operator-field.hpp"

namespace ceed {
  namespace occa {
    typedef std::vector<OperatorField> OperatorFieldVector;

    class OperatorArgs : public QFunctionArgs {
     public:
      OperatorFieldVector opInputs;
      OperatorFieldVector opOutputs;

      OperatorArgs();
      OperatorArgs(CeedOperator op);

      void setupArgs(CeedOperator op);

      const OperatorField& getOpField(const bool isInput,
                                      const int index) const;

      const OperatorField& getOpInput(const int index) const;

      const OperatorField& getOpOutput(const int index) const;
    };
  }
}

#endif
