SUBROUTINE get_gadget_file(iunit,part_file,ifile,npart_in_file,pos,vel)
  USE globalvars
  implicit none
  CHARACTER*180 :: part_file
  INTEGER :: ifile,iunit
  real*8::box_dp
  real,dimension(3,npart_in_file)::pos,vel
  integer::npart_in_file
  CHARACTER*20 :: conv='LITTLE_ENDIAN'
  INTEGER, DIMENSION(6) :: npart,npartot
  DOUBLE PRECISION, DIMENSION(6) :: masses
  DOUBLE PRECISION :: h,Lambda,Om,aexpn,redshift
  INTEGER :: n_o_b,SF,Fb,nFiles,Cool
  CHARACTER*60 :: c5
  CHARACTER*4 :: header




  OPEN(iunit, FILE=part_file,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  READ(iunit)
  
  ! Reading the parameters line
  
  READ(iunit) npart(1:6),masses(1:6),aexpn,redshift,SF,Fb,npartot(1:6),Cool, &
       & nFiles,box_dp,Om,Lambda,h,c5
  
     
  

  
  
 
  READ(iunit) header,n_o_b
  !write(*,*) header,'H',n_o_b
  READ(iunit) pos
  !PRINT *,' Positions read from file ',ifile,pos(:,1)
  ! Velocities
  READ(iunit) header,n_o_b
  !write(*,*) header,'V',n_o_b
  READ(iunit) vel
  !PRINT *,' Velocities read from file ',ifile

  close(iunit)
  
  pos=pos/1000.



END SUBROUTINE get_gadget_file

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


