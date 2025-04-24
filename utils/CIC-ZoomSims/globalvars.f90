module globalvars
  implicit none
  
  !for read gadget
  real*8::mpart,boxsize

  real*8,allocatable::cic_vx(:,:,:),cic_vy(:,:,:),cic_vz(:,:,:),cic_grid(:,:,:)
  real*8,allocatable::count_vx(:,:,:),count_vy(:,:,:),count_vz(:,:,:),count_grid(:,:,:)
  integer::ncells


  integer*8::ntot

!!$  LOGICAL :: new_format=.FALSE.
!!$  CHARACTER*4 :: header
!!$  CHARACTER*20 :: conv='BIG_ENDIAN'
!!$  CHARACTER*60 :: c5
!!$  INTEGER :: n_o_b,SF,Fb,nFiles,Cool
!!$  INTEGER*8, ALLOCATABLE :: id(:),omitted(:)
!!$  INTEGER, DIMENSION(6) :: npart,npartot
!!$  DOUBLE PRECISION :: h,Lambda,Om,aexpn,redshift,boxsize
!!$!  real, ALLOCATABLE :: pos(:,:),vel(:,:)
!!$  real*8,allocatable::mass(:)
!!$  DOUBLE PRECISION, DIMENSION(6) :: masses

!  integer::npart_in_file

end module globalvars

