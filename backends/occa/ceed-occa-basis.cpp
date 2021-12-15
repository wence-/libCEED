#include "ceed-occa-basis.hpp"
#include "ceed-occa-tensor-basis.hpp"

namespace ceed {
  namespace occa {
    Basis::Basis() :
        ceedComponentCount(0),
        dim(0),
        P(0),
        Q(0) {}

    Basis::~Basis() {}

    Basis* Basis::getBasis(CeedBasis basis,
                           const bool assertValid) {
      if (!basis) {
        return NULL;
      }

      int ierr;
      Basis *basis_ = NULL;

      ierr = CeedBasisGetData(basis, &basis_);
      if (assertValid) {
        CeedOccaFromChk(ierr);
      }

      return basis_;
    }

    Basis* Basis::from(CeedBasis basis) {
      Basis *basis_ = getBasis(basis);
      if (!basis_) {
        return NULL;
      }

      int ierr;
      ierr = basis_->setCeedFields(basis); CeedOccaFromChk(ierr);

      return basis_;
    }

    Basis* Basis::from(CeedOperatorField operatorField) {
      int ierr;
      CeedBasis basis;
      ierr = CeedOperatorFieldGetBasis(operatorField, &basis); CeedOccaFromChk(ierr);
      return from(basis);
    }

    int Basis::setCeedFields(CeedBasis basis) {
      int ierr;

      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basis, &ceedComponentCount); CeedChk(ierr);

      return CEED_ERROR_SUCCESS;
    }

    //---[ Ceed Callbacks ]-----------
    int Basis::registerCeedFunction(Ceed ceed, CeedBasis basis,
                                    const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Basis", basis, fname, f);
    }

    int Basis::ceedApply(CeedBasis basis, const CeedInt nelem,
                         CeedTransposeMode tmode,
                         CeedEvalMode emode, CeedVector u, CeedVector v) {
      Basis *basis_ = Basis::from(basis);
      Vector *U = Vector::from(u);
      Vector *V = Vector::from(v);

      if (!basis_) {
        return staticCeedError("Incorrect CeedBasis argument: op");
      }

      return basis_->apply(
        nelem,
        tmode, emode,
        U, V
      );
    }

    int Basis::ceedDestroy(CeedBasis basis) {
      delete getBasis(basis, false);
      return CEED_ERROR_SUCCESS;
    }
  }
}
