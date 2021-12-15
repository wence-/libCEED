#include "ceed-occa-operator-args.hpp"

namespace ceed {
  namespace occa {
    OperatorArgs::OperatorArgs() :
        QFunctionArgs() {}

    OperatorArgs::OperatorArgs(CeedOperator op) :
        QFunctionArgs() {

      setupArgs(op);
    }

    void OperatorArgs::setupArgs(CeedOperator op) {
      CeedQFunction qf;
      CeedOperatorField *ceedInputFields, *ceedOutputFields;
      int ierr = 0;

      ierr = CeedOperatorGetQFunction(op, &qf);
      CeedOccaValidChk(_isValid, ierr);
      setupQFunctionArgs(qf);

      if (!_isValid) {
        return;
      }

      ierr = CeedOperatorGetFields(op, NULL, &ceedInputFields, NULL, &ceedOutputFields);
      CeedOccaValidChk(_isValid, ierr);

      for (int i = 0; i < _inputCount; ++i) {
        OperatorField field = OperatorField(ceedInputFields[i]);
        opInputs.push_back(field);
        _isValid &= field.isValid();
      }

      for (int i = 0; i < _outputCount; ++i) {
        OperatorField field = OperatorField(ceedOutputFields[i]);
        opOutputs.push_back(field);
        _isValid &= field.isValid();
      }
    }

    const OperatorField& OperatorArgs::getOpField(const bool isInput,
                                                  const int index) const {
      return isInput ? opInputs[index] : opOutputs[index];
    }

    const OperatorField& OperatorArgs::getOpInput(const int index) const {
      return opInputs[index];
    }

    const OperatorField& OperatorArgs::getOpOutput(const int index) const {
      return opOutputs[index];
    }
  }
}
