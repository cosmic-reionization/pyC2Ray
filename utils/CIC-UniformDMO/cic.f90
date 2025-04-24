program cic
  use omp_lib
  use globalvars
  implicit none
  character*180::fname,fcic,nc,fbase,fcount,ferr
  character*100::sim
  integer::ifile,ithread,n_p
  integer*8::ntot_temp
  integer::nthreads,iargc,isnap
  real*8::start,finish,startall,finishall
  CHARACTER*20 :: conv='LITTLE_ENDIAN'
  INTEGER, DIMENSION(6) :: npart,npartot
  DOUBLE PRECISION :: h,Lambda,Om,aexpn,redshift
  CHARACTER*4 :: header
  INTEGER :: n_o_b,SF,Fb,nFiles,Cool
  DOUBLE PRECISION, DIMENSION(6) :: masses
  CHARACTER*60 :: c5,snc
  character*100::a,b,c





  if (iargc().eq.4) then
     call getarg(1,sim)
     call getarg(2,a)
     call getarg(3,b)
     call getarg(4,c)

     read(a,*) isnap
     read(b,*) n_p
     read(c,*) ncells
  else if (iargc().ne.1) then
     write(*,*) 'input error: SIM ISNAP NPART NCELL'
     stop
  end if

!inputs required by the user, sim path, snap number, particles per dimension, number of cells 
  write(*,*) sim
  write(*,*) isnap
  write(*,*) n_p
  write(*,*) ncells



  nthreads=8


  ntot=int8(n_p)**3
  write(nc,*) ncells

  
  !write(*,*) isnap,a


  write(snc,*) isnap  
  if (isnap.le.99) then
     snc='0'//trim(adjustl(snc))
  else
     snc=trim(adjustl(snc))
  end if



!for ESMD
  fname='/store/clues/HESTIA/FULL_BOX/1024/DM_ONLY/37_11/output/snapdir_'//trim(adjustl(snc))//'/snapshot_'//trim(adjustl(snc))//'.0'
  fbase='/store/clues/HESTIA/FULL_BOX/1024/DM_ONLY/37_11/output/snapdir_'//trim(adjustl(snc))//'/snapshot_'//trim(adjustl(snc))//'.'
  
  !Save paths (fcic= cic path, fcoun= count in cell output, ferr= error file path)
  fcic='./'//trim(adjustl(snc))//'.BCIC.'//trim(adjustl(nc))
  fcount='./'//trim(adjustl(snc))//'.BCOUNT.'//trim(adjustl(nc))
  ferr='./'//trim(adjustl(snc))//'.ERROR.'//trim(adjustl(nc))


  write(*,*) fname,fbase,fcic,fcount


  allocate(cic_grid(ncells,ncells,ncells),cic_vx(ncells,ncells,ncells),cic_vy(ncells,ncells,ncells),cic_vz(ncells,ncells,ncells))
  allocate(count_grid(ncells,ncells,ncells),count_vx(ncells,ncells,ncells),count_vy(ncells,ncells,ncells),count_vz(ncells,ncells,ncells))



  ! Opening file and reading the header info for each file
  OPEN(2, FILE=fname,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  READ(2) npart(1:6),masses(1:6),aexpn,redshift,SF,Fb,npartot(1:6),Cool, &
       & nFiles,boxsize,Om,Lambda,h,c5

  boxsize=boxsize/1000.
  mpart=masses(2)!*1.E10
 
  !ntot=int8(sum(npartot))


  WRITE(*,*) 'Parameter line read'
  WRITE(*,*) 'Number of files: ', nFiles
  WRITE(*,*) 'Total number of particles in the simulation: ', ntot
  WRITE(*,*) 'Total number of particles in file: ', npart
  WRITE(*,*) 'Masses: ',  masses
  write(*,*) 'Boxsize',boxsize
  write(*,*) 'Redshift',redshift,'expn',aexpn
  CLOSE(2)


  cic_grid(:,:,:)=0.

  count_grid(:,:,:)=0.
   
 
  call omp_set_num_threads(nthreads)

  startall=omp_get_wtime()
  start=omp_get_wtime()   


  ntot_temp=0

  !$OMP PARALLEL DO private(ithread) shared(fbase,cic_grid,cic_vx,cic_vy,cic_vz,count_vx,count_vy,count_vz,count_grid,nthreads,nfiles)
  DO ithread=0,nthreads-1     
     call cic_dens(fbase,ithread,nthreads,nfiles)
     finish=omp_get_wtime()
     write(*,*)finish-start,'elasped time on thread',ithread
     start=omp_get_wtime()

  end DO
  !$OMP END PARALLEL DO

  finishall=omp_get_wtime()


  write(*,*) finishall-startall,' elapsed time for all files'



  call write_out_cic(fcic,fcount,ferr)



 


end program cic

