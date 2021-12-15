#ifndef CEED_OCCA_OPERATORFIELD_HEADER
#define CEED_OCCA_OPERATORFIELD_HEADER

#include "ceed-occa-context.hpp"

namespace ceed {
  namespace occa {
    class Basis;
    class ElemRestriction;
    class Vector;

    class OperatorField {
     private:
      bool _isValid;
      bool _usesActiveVector;

     public:
      Vector *vec;
      Basis *basis;
      ElemRestriction *elemRestriction;

      OperatorField(CeedOperatorField opField);

      bool isValid() const;

      //---[ Vector Info ]--------------
      bool usesActiveVector() const;
      //================================

      //---[ Basis Info ]---------------
      bool hasBasis() const;
      int usingTensorBasis() const;

      int getComponentCount() const;
      int getP() const;
      int getQ() const;
      int getDim() const;
      //================================

      //---[ ElemRestriction Info ]-----
      int getElementCount() const;
      int getElementSize() const;
      //================================
    };
  }
}

#endif
