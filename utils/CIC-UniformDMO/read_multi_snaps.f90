subroutine read_multi_snaps(part_file,fbase)
  use globalvars
  implicit none
  character*180::part_file,fbase
  character*4::nfile_str
  integer::ifile



  


  OPEN(2, FILE=part_file,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  READ(2) header, n_o_b
  WRITE(*,*) header
  
  IF ((n_o_b .EQ. 264) .OR. (n_o_b .EQ. 134283264)) THEN
     new_format = .TRUE.
     IF (n_o_b .EQ. 134283264) conv='LITTLE_ENDIAN'
  ENDIF
  CLOSE(2)
  
  WRITE(*,*) 'New format = ',new_format
  ! Reading the parameters line
  
  OPEN(2, FILE=part_file,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  IF (new_format) READ(2)
  READ(2) npart(1:6),masses(1:6),aexpn,redshift,SF,Fb,npartot(1:6),Cool, &
       & nFiles,boxsize,Om,Lambda,h,c5

  boxsize=boxsize/1000.




  WRITE(*,*) 'Parameter line read'
  WRITE(*,*) 'Number of files: ', nFiles
  WRITE(*,*) 'Total number of particles in the simulation: ', sum(npartot)
  WRITE(*,*) 'Total number of particles in file: ', npart
  WRITE(*,*) 'Masses: ', masses*1.E10
  write(*,*) 'Boxsize',boxsize
  write(*,*) 'Redshift',redshift,'expn',aexpn
  CLOSE(2)


  ntot=8589934592


  ALLOCATE(pos(3,ntot),vel(3,ntot),id(ntot),mass(ntot))




  DO ifile=0,nFiles-1
     write(*,*) 'read file',ifile,' of ',nfiles
     
     WRITE(nfile_str,'(i4)') ifile
     nfile_str=ADJUSTL(nfile_str)
     
    
     part_file=trim(adjustl(fbase))//trim(adjustl(nfile_str))
     
     print *,'Reading from file ',part_file
      
     

     CALL get_gadget_file(part_file,ifile)

  

  end DO


  



 


  write(*,*) 'done reading snapshot'

end subroutine read_multi_snaps
