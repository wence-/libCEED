#include "ceed-occa-operator-field.hpp"
#include "ceed-occa-basis.hpp"
#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
  namespace occa {
    OperatorField::OperatorField(CeedOperatorField opField) :
        _isValid(false),
        _usesActiveVector(false),
        vec(NULL),
        basis(NULL),
        elemRestriction(NULL) {

      CeedBasis ceedBasis;
      CeedVector ceedVector;
      CeedElemRestriction ceedElemRestriction;
      int ierr = 0;

      ierr = CeedOperatorFieldGetBasis(opField, &ceedBasis);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetVector(opField, &ceedVector);
      CeedOccaValidChk(_isValid, ierr);

      ierr = CeedOperatorFieldGetElemRestriction(opField, &ceedElemRestriction);
      CeedOccaValidChk(_isValid, ierr);

      _isValid = true;
      _usesActiveVector = ceedVector == CEED_VECTOR_ACTIVE;

      vec = Vector::from(ceedVector);
      basis = Basis::from(ceedBasis);
      elemRestriction = ElemRestriction::from(ceedElemRestriction);
    }

    bool OperatorField::isValid() const {
      return _isValid;
    }

    //---[ Vector Info ]----------------
    bool OperatorField::usesActiveVector() const {
      return _usesActiveVector;
    }
    //==================================

    //---[ Basis Info ]-----------------
    bool OperatorField::hasBasis() const {
      return basis;
    }

    int OperatorField::usingTensorBasis() const {
      return basis->isTensorBasis();
    }

    int OperatorField::getComponentCount() const {
      return (
        basis
        ? basis->ceedComponentCount
        : 1
      );
    }

    int OperatorField::getP() const {
      return (
        basis
        ? basis->P
        : 0
      );
    }

    int OperatorField::getQ() const {
      return (
        basis
        ? basis->Q
        : 0
      );
    }

    int OperatorField::getDim() const {
      return (
        basis
        ? basis->dim
        : 1
      );
    }
    //==================================

    //---[ ElemRestriction Info ]-------
    int OperatorField::getElementCount() const {
      return (
        elemRestriction
        ? elemRestriction->ceedElementCount
        : 1
      );
    }

    int OperatorField::getElementSize() const {
      return (
        elemRestriction
        ? elemRestriction->ceedElementSize
        : 1
      );
    }
    //==================================
  }
}
