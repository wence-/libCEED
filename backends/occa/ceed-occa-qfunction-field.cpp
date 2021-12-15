#include "ceed-occa-qfunction-field.hpp"

namespace ceed {
  namespace occa {
    QFunctionField::QFunctionField(CeedQFunctionField qfField) :
        _isValid(false),
        size(0) {

      int ierr = 0;

      ierr = CeedQFunctionFieldGetEvalMode(qfField, &evalMode);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedQFunctionFieldGetSize(qfField, &size);
      CeedOccaValidChk(_isValid, ierr);

      _isValid = true;
    }

    bool QFunctionField::isValid() const {
      return _isValid;
    }
  }
}
