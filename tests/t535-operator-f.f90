!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't535-operator-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j,k
      integer stridesu(3),stridesqd(3)
      integer erestrictx,erestrictu,erestrictui,erestrictqi
      integer bx,bu
      integer qf_setup_mass,qf_setup_diff,qf_apply
      integer op_setup_mass,op_setup_diff,op_apply
      integer qdata_mass,qdata_diff,x,a,u,v
      integer nelem,p,q,d
      integer row,col,offset
      parameter(nelem=6)
      parameter(p=3)
      parameter(q=4)
      parameter(d=2)
      integer ndofs,nqpts,nx,ny
      parameter(nx=3)
      parameter(ny=2)
      parameter(ndofs=(nx*2+1)*(ny*2+1))
      parameter(nqpts=nelem*q*q)
      integer indx(nelem*p*p)
      real*8 arrx(d*ndofs),aa(nqpts),uu(ndofs),vv(ndofs),atrue(ndofs)
      integer*8 xoffset,aoffset,uoffset,voffset

      character arg*32

      external setup_mass,setup_diff,apply,apply_lin

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

! DoF Coordinates
      do i=0,nx*2
        do j=0,ny*2
          arrx(i+j*(nx*2+1)+0*ndofs+1)=1.d0*i/(2*nx)
          arrx(i+j*(nx*2+1)+1*ndofs+1)=1.d0*j/(2*ny)
        enddo
      enddo
      call ceedvectorcreate(ceed,d*ndofs,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)

! Qdata Vector
      call ceedvectorcreate(ceed,nqpts,qdata_mass,err)
      call ceedvectorcreate(ceed,nqpts*d*(d+1)/2,qdata_diff,err)

! Element Setup
      do i=0,nelem-1
        col=mod(i,nx)
        row=i/nx
        offset=col*(p-1)+row*(nx*2+1)*(p-1)
        do j=0,p-1
          do k=0,p-1
            indx(p*(p*i+k)+j+1)=offset+k*(nx*2+1)+j
          enddo
        enddo
      enddo

! Restrictions
      call ceedelemrestrictioncreate(ceed,nelem,p*p,d,ndofs,d*ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictx,err)

      call ceedelemrestrictioncreate(ceed,nelem,p*p,1,1,ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictu,err)
      stridesu=[1,q*q,q*q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*q,1,nqpts,&
     & stridesu,erestrictui,err)

      stridesqd=[1,q*q,q*q*d*(d+1)/2]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*q,d*(d+1)/2,&
     & d*(d+1)/2*nqpts,stridesqd,erestrictqi,err)

! Bases
      call ceedbasiscreatetensorh1lagrange(ceed,d,d,p,q,ceed_gauss,bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,d,1,p,q,ceed_gauss,bu,err)

! QFunction - setup mass
      call ceedqfunctioncreateinterior(ceed,1,setup_mass,&
     &SOURCE_DIR&
     &//'t532-operator.h:setup_mass'//char(0),qf_setup_mass,err)
      call ceedqfunctionaddinput(qf_setup_mass,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup_mass,'weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup_mass,'qdata',1,ceed_eval_none,err)

! Operator - setup mass
      call ceedoperatorcreate(ceed,qf_setup_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup_mass,err)
      call ceedoperatorsetfield(op_setup_mass,'dx',erestrictx,&
     & bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup_mass,'weight',&
     & ceed_elemrestriction_none,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup_mass,'qdata',erestrictui,&
     & ceed_basis_collocated,ceed_vector_active,err)

! QFunction - setup diff
      call ceedqfunctioncreateinterior(ceed,1,setup_diff,&
     &SOURCE_DIR&
     &//'t532-operator.h:setup_diff'//char(0),qf_setup_diff,err)
      call ceedqfunctionaddinput(qf_setup_diff,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup_diff,'weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup_diff,'qdata',&
     & d*(d+1)/2,ceed_eval_none,err)

! Operator - setup diff
      call ceedoperatorcreate(ceed,qf_setup_diff,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup_diff,err)
      call ceedoperatorsetfield(op_setup_diff,'dx',erestrictx,&
     & bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup_diff,'weight',&
     & ceed_elemrestriction_none,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup_diff,'qdata',erestrictqi,&
     & ceed_basis_collocated,ceed_vector_active,err)

! Apply Setup Operators
      call ceedoperatorapply(op_setup_mass,x,qdata_mass,&
     & ceed_request_immediate,err)
      call ceedoperatorapply(op_setup_diff,x,qdata_diff,&
     & ceed_request_immediate,err)

! QFunction - apply
      call ceedqfunctioncreateinterior(ceed,1,apply,&
     &SOURCE_DIR&
     &//'t532-operator.h:apply'//char(0),qf_apply,err)
      call ceedqfunctionaddinput(qf_apply,'du',d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_apply,'mass qdata',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_apply,'diff qdata',&
     & d*(d+1)/2,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_apply,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_apply,'v',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_apply,'dv',d,ceed_eval_grad,err)

! Operator - apply
      call ceedoperatorcreate(ceed,qf_apply,ceed_qfunction_none,&
     & ceed_qfunction_none,op_apply,err)
      call ceedoperatorsetfield(op_apply,'du',erestrictu,&
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_apply,'mass qdata',erestrictui,&
     & ceed_basis_collocated,qdata_mass,err)
      call ceedoperatorsetfield(op_apply,'diff qdata',erestrictqi,&
     & ceed_basis_collocated,qdata_diff,err)
      call ceedoperatorsetfield(op_apply,'u',erestrictu,&
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_apply,'v',erestrictu,&
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_apply,'dv',erestrictu,&
     & bu,ceed_vector_active,err)

! Assemble Diagonal
      call ceedvectorcreate(ceed,ndofs,a,err)
      call ceedoperatorlinearassemblediagonal(op_apply,a,&
     & ceed_request_immediate,err)

! Manually assemble diagonal
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,0.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      do i=1,ndofs
        call ceedvectorgetarray(u,ceed_mem_host,uu,uoffset,err)
        uu(i+uoffset)=1.d0
        if (i>1) then
          uu(i-1+uoffset)=0.d0
        endif
        call ceedvectorrestorearray(u,uu,uoffset,err)

        call ceedoperatorapply(op_apply,u,v,ceed_request_immediate,err)

        call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
        atrue(i)=vv(voffset+i)
        call ceedvectorrestorearrayread(v,vv,voffset,err)
      enddo

! Check Output
      call ceedvectorgetarrayread(a,ceed_mem_host,aa,aoffset,err)
      do i=1,ndofs
        if (abs(aa(aoffset+i)-atrue(i))>1.0d-14) then
! LCOV_EXCL_START
          write(*,*) '[',i,'] Error in assembly: ',aa(aoffset+i),' != ',&
     &      atrue(i)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(a,aa,aoffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setup_mass,err)
      call ceedqfunctiondestroy(qf_setup_diff,err)
      call ceedqfunctiondestroy(qf_apply,err)
      call ceedoperatordestroy(op_setup_mass,err)
      call ceedoperatordestroy(op_setup_diff,err)
      call ceedoperatordestroy(op_apply,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictqi,err)
      call ceedbasisdestroy(bu,err)
      call ceedbasisdestroy(bx,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(a,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedvectordestroy(qdata_mass,err)
      call ceedvectordestroy(qdata_diff,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
