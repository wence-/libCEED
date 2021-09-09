// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative

//! A Ceed Basis defines the discrete finite element basis and associated
//! quadrature rule.

use crate::prelude::*;

// -----------------------------------------------------------------------------
// CeedBasis option
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum BasisOpt<'a> {
    Some(&'a Basis<'a>),
    Collocated,
}
/// Construct a BasisOpt reference from a Basis reference
impl<'a> From<&'a Basis<'_>> for BasisOpt<'a> {
    fn from(basis: &'a Basis) -> Self {
        debug_assert!(basis.ptr != unsafe { bind_ceed::CEED_BASIS_COLLOCATED });
        Self::Some(basis)
    }
}
impl<'a> BasisOpt<'a> {
    /// Transform a Rust libCEED BasisOpt into C libCEED CeedBasis
    pub(crate) fn to_raw(self) -> bind_ceed::CeedBasis {
        match self {
            Self::Some(basis) => basis.ptr,
            Self::Collocated => unsafe { bind_ceed::CEED_BASIS_COLLOCATED },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedBasis field option
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum BasisFieldOpt<'a> {
    Some(Basis<'a>),
    Collocated,
}
impl<'a> BasisFieldOpt<'a> {
    /// Check if a BasisFieldOpt is Some
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild").unwrap();
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)
    ///     .unwrap();
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed
    ///     .strided_elem_restriction(ne, 2, 1, q * ne, strides)
    ///     .unwrap();
    ///
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)
    ///     .unwrap()
    ///     .field("dx", &r, &b, VectorOpt::Active)
    ///     .unwrap()
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)
    ///     .unwrap()
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)
    ///     .unwrap();
    ///
    /// let inputs = op.inputs().unwrap();
    ///
    /// assert!(inputs[0].basis().is_some(), "Incorrect field Basis");
    /// assert!(inputs[1].basis().is_some(), "Incorrect field Basis");
    /// ```
    pub fn is_some(self) -> bool {
        match self {
            Self::Some(_) => true,
            Self::Collocated => false,
        }
    }

    /// Check if a BasisFieldOpt is Active
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild").unwrap();
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)
    ///     .unwrap();
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed
    ///     .strided_elem_restriction(ne, 2, 1, q * ne, strides)
    ///     .unwrap();
    ///
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)
    ///     .unwrap()
    ///     .field("dx", &r, &b, VectorOpt::Active)
    ///     .unwrap()
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)
    ///     .unwrap()
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)
    ///     .unwrap();
    ///
    /// let outputs = op.outputs().unwrap();
    ///
    /// assert!(outputs[0].basis().is_collocated(), "Incorrect field Basis");
    /// ```
    pub fn is_collocated(self) -> bool {
        match self {
            Self::Some(_) => false,
            Self::Collocated => true,
        }
    }

    /// Get the Basis for a BasisFieldOpt
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild").unwrap();
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)
    ///     .unwrap();
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed
    ///     .strided_elem_restriction(ne, 2, 1, q * ne, strides)
    ///     .unwrap();
    ///
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)
    ///     .unwrap()
    ///     .field("dx", &r, &b, VectorOpt::Active)
    ///     .unwrap()
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)
    ///     .unwrap()
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)
    ///     .unwrap();
    ///
    /// let inputs = op.inputs().unwrap();
    ///
    /// assert!(inputs[0].basis().basis().is_ok(), "Incorrect field Basis");
    /// assert!(inputs[1].basis().basis().is_ok(), "Incorrect field Basis");
    ///
    /// let outputs = op.outputs().unwrap();
    ///
    /// assert!(outputs[0].basis().basis().is_err(), "Incorrect field Basis");
    /// ```
    pub fn basis(self) -> crate::Result<Basis<'a>> {
        match self {
            Self::Some(basis) => Ok(basis),
            Self::Collocated => Err(crate::CeedError {
                message: "Collocated BasisFieldOpt has no Basis".to_string(),
            }),
        }
    }

    pub fn unwrap(self) -> Basis<'a> {
        match self {
            Self::Some(basis) => basis,
            Self::Collocated => panic!("Collocated BasisFieldOpt has no Basis"),
        }
    }
}

// -----------------------------------------------------------------------------
// CeedBasis context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct Basis<'a> {
    ceed: &'a crate::Ceed,
    pub(crate) ptr: bind_ceed::CeedBasis,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for Basis<'a> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_BASIS_COLLOCATED {
                bind_ceed::CeedBasisDestroy(&mut self.ptr);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for Basis<'a> {
    /// View a Basis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(1, 2, 3, 4, QuadMode::Gauss)
    ///     .unwrap();
    /// println!("{}", b);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedBasisView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl<'a> Basis<'a> {
    // Constructors
    pub fn create_tensor_H1(
        ceed: &'a crate::Ceed,
        dim: usize,
        ncomp: usize,
        P1d: usize,
        Q1d: usize,
        interp1d: &[crate::Scalar],
        grad1d: &[crate::Scalar],
        qref1d: &[crate::Scalar],
        qweight1d: &[crate::Scalar],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P1d, Q1d) = (
            i32::try_from(dim).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(P1d).unwrap(),
            i32::try_from(Q1d).unwrap(),
        );
        let ierr = unsafe {
            bind_ceed::CeedBasisCreateTensorH1(
                ceed.ptr,
                dim,
                ncomp,
                P1d,
                Q1d,
                interp1d.as_ptr(),
                grad1d.as_ptr(),
                qref1d.as_ptr(),
                qweight1d.as_ptr(),
                &mut ptr,
            )
        };
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    pub fn create_tensor_H1_Lagrange(
        ceed: &'a crate::Ceed,
        dim: usize,
        ncomp: usize,
        P: usize,
        Q: usize,
        qmode: crate::QuadMode,
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P, Q, qmode) = (
            i32::try_from(dim).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(P).unwrap(),
            i32::try_from(Q).unwrap(),
            qmode as bind_ceed::CeedQuadMode,
        );
        let ierr = unsafe {
            bind_ceed::CeedBasisCreateTensorH1Lagrange(ceed.ptr, dim, ncomp, P, Q, qmode, &mut ptr)
        };
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    pub fn create_H1(
        ceed: &'a crate::Ceed,
        topo: crate::ElemTopology,
        ncomp: usize,
        nnodes: usize,
        nqpts: usize,
        interp: &[crate::Scalar],
        grad: &[crate::Scalar],
        qref: &[crate::Scalar],
        qweight: &[crate::Scalar],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (topo, ncomp, nnodes, nqpts) = (
            topo as bind_ceed::CeedElemTopology,
            i32::try_from(ncomp).unwrap(),
            i32::try_from(nnodes).unwrap(),
            i32::try_from(nqpts).unwrap(),
        );
        let ierr = unsafe {
            bind_ceed::CeedBasisCreateH1(
                ceed.ptr,
                topo,
                ncomp,
                nnodes,
                nqpts,
                interp.as_ptr(),
                grad.as_ptr(),
                qref.as_ptr(),
                qweight.as_ptr(),
                &mut ptr,
            )
        };
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    pub(crate) fn from_raw(
        ceed: &'a crate::Ceed,
        ptr: bind_ceed::CeedBasis,
    ) -> crate::Result<Self> {
        Ok(Self { ceed, ptr })
    }

    /// Apply basis evaluation from nodes to quadrature points or vice versa
    ///
    /// * `nelem` - The number of elements to apply the basis evaluation to
    /// * `tmode` - `TrasposeMode::NoTranspose` to evaluate from nodes to
    ///               quadrature points, `TransposeMode::Transpose` to apply the
    ///               transpose, mapping from quadrature points to nodes
    /// * `emode` - `EvalMode::None` to use values directly, `EvalMode::Interp`
    ///               to use interpolated values, `EvalMode::Grad` to use
    ///               gradients, `EvalMode::Weight` to use quadrature weights
    /// * `u`     - Input Vector
    /// * `v`     - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// const Q: usize = 6;
    /// let bu = ceed
    ///     .basis_tensor_H1_Lagrange(1, 1, Q, Q, QuadMode::GaussLobatto)
    ///     .unwrap();
    /// let bx = ceed
    ///     .basis_tensor_H1_Lagrange(1, 1, 2, Q, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// let x_corners = ceed.vector_from_slice(&[-1., 1.]).unwrap();
    /// let mut x_qpts = ceed.vector(Q).unwrap();
    /// let mut x_nodes = ceed.vector(Q).unwrap();
    /// bx.apply(
    ///     1,
    ///     TransposeMode::NoTranspose,
    ///     EvalMode::Interp,
    ///     &x_corners,
    ///     &mut x_nodes,
    /// );
    /// bu.apply(
    ///     1,
    ///     TransposeMode::NoTranspose,
    ///     EvalMode::Interp,
    ///     &x_nodes,
    ///     &mut x_qpts,
    /// );
    ///
    /// // Create function x^3 + 1 on Gauss Lobatto points
    /// let mut u_arr = [0.; Q];
    /// u_arr
    ///     .iter_mut()
    ///     .zip(x_nodes.view().iter())
    ///     .for_each(|(u, x)| *u = x * x * x + 1.);
    /// let u = ceed.vector_from_slice(&u_arr).unwrap();
    ///
    /// // Map function to Gauss points
    /// let mut v = ceed.vector(Q).unwrap();
    /// v.set_value(0.);
    /// bu.apply(1, TransposeMode::NoTranspose, EvalMode::Interp, &u, &mut v)
    ///     .unwrap();
    ///
    /// // Verify results
    /// v.view()
    ///     .iter()
    ///     .zip(x_qpts.view().iter())
    ///     .for_each(|(v, x)| {
    ///         let true_value = x * x * x + 1.;
    ///         assert!(
    ///             (*v - true_value).abs() < 10.0 * libceed::EPSILON,
    ///             "Incorrect basis application"
    ///         );
    ///     });
    /// ```
    pub fn apply(
        &self,
        nelem: usize,
        tmode: TransposeMode,
        emode: EvalMode,
        u: &Vector,
        v: &mut Vector,
    ) -> crate::Result<i32> {
        let (nelem, tmode, emode) = (
            i32::try_from(nelem).unwrap(),
            tmode as bind_ceed::CeedTransposeMode,
            emode as bind_ceed::CeedEvalMode,
        );
        let ierr =
            unsafe { bind_ceed::CeedBasisApply(self.ptr, nelem, tmode, emode, u.ptr, v.ptr) };
        self.ceed.check_error(ierr)
    }

    /// Returns the dimension for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let dim = 2;
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(dim, 1, 3, 4, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// let d = b.dimension();
    /// assert_eq!(d, dim, "Incorrect dimension");
    /// ```
    pub fn dimension(&self) -> usize {
        let mut dim = 0;
        unsafe { bind_ceed::CeedBasisGetDimension(self.ptr, &mut dim) };
        usize::try_from(dim).unwrap()
    }

    /// Returns number of components for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let ncomp = 2;
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(1, ncomp, 3, 4, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// let n = b.num_components();
    /// assert_eq!(n, ncomp, "Incorrect number of components");
    /// ```
    pub fn num_components(&self) -> usize {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedBasisGetNumComponents(self.ptr, &mut ncomp) };
        usize::try_from(ncomp).unwrap()
    }

    /// Returns total number of nodes (in dim dimensions) of a CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let p = 3;
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(2, 1, p, 4, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// let nnodes = b.num_nodes();
    /// assert_eq!(nnodes, p * p, "Incorrect number of nodes");
    /// ```
    pub fn num_nodes(&self) -> usize {
        let mut nnodes = 0;
        unsafe { bind_ceed::CeedBasisGetNumNodes(self.ptr, &mut nnodes) };
        usize::try_from(nnodes).unwrap()
    }

    /// Returns total number of quadrature points (in dim dimensions) of a
    /// CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let q = 4;
    /// let b = ceed
    ///     .basis_tensor_H1_Lagrange(2, 1, 3, q, QuadMode::Gauss)
    ///     .unwrap();
    ///
    /// let nqpts = b.num_quadrature_points();
    /// assert_eq!(nqpts, q * q, "Incorrect number of quadrature points");
    /// ```
    pub fn num_quadrature_points(&self) -> usize {
        let mut Q = 0;
        unsafe {
            bind_ceed::CeedBasisGetNumQuadraturePoints(self.ptr, &mut Q);
        }
        usize::try_from(Q).unwrap()
    }
}

// -----------------------------------------------------------------------------
