program cic
  use omp_lib
  use globalvars
  implicit none
  character*180::fname,fcic,nc,fbase,fcount
  integer::ifile,ithread
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
  character*100::a


  isnap=190
  
  nthreads=12





!  ncells=64 ! 1Mpc cells
!  ncells=32 ! 2Mpc cells
  ncells=16 ! 4.0Mpc cells
  !ncells=10 ! 6.4Mpc
  !ncells=9  ! 7.1 Mpc
!  ncells=8  ! 8Mpc cells
  !ncells=7  ! 9.1 Mpc
  !ncells=6  ! 10.6 Mpc
!  ncells=4  ! 16Mpc cells

  write(nc,*) ncells

  allocate(counts_grid(ncells,ncells,ncells))




  write(snc,*) isnap  
  if (isnap.le.99) then
     snc='0'//trim(adjustl(snc))
  else
     snc=trim(adjustl(snc))
  end if



!!$
!!$  fname='/store/clues01/B64_WM5_10909/BOX_DM/2048/SNAPS/snapdir_'//trim(adjustl(snc))//'/snap_'//trim(adjustl(snc))//'.0'
!!$  fbase='/store/clues01/B64_WM5_10909/BOX_DM/2048/SNAPS/snapdir_'//trim(adjustl(snc))//'/snap_'//trim(adjustl(snc))//'.'
!!$  fcic='/store/dpk01/nil/10909/CIC-OMP/2048_dm_WM5_'//trim(adjustl(snc))//'.BCIC.'//trim(adjustl(nc))
!!$  fcount='/store/dpk01/nil/10909/CIC-OMP/2048_dm_WM5_'//trim(adjustl(snc))//'.BCOUNT.'//trim(adjustl(nc))
!!$  
!!$  
!!$
!!$ ntot=int8(2048)**3


  fname='/store/clues/B64_WM5_10909/BOX_DM/1024/SNAPS/snapdir_190/snap_190.0'
  fbase='/store/clues/B64_WM5_10909/BOX_DM/1024/SNAPS/snapdir_190/snap_190.'
  fcic='/store/erebos/nil/10909/CIC-HaloEfficiency/1024_dm_WM5_190.BCIC.'//trim(adjustl(nc))
  fcount='/store/erebos/nil/10909/CIC-HaloEfficiency/1024_dm_WM5_190.BCOUNT.'//trim(adjustl(nc))
  ntot=int8(1024)**3

  write(*,*) fname,fbase,fcic,fcount



  OPEN(2, FILE=fname,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  READ(2) header, n_o_b
  
 
  CLOSE(2)
  
  ! Reading the parameters line
  
  OPEN(2, FILE=fname,FORM='unformatted',STATUS='OLD',CONVERT=conv)
  READ(2)
  READ(2) npart(1:6),masses(1:6),aexpn,redshift,SF,Fb,npartot(1:6),Cool, &
       & nFiles,boxsize,Om,Lambda,h,c5

  boxsize=boxsize/1000.
  mpart=masses(2)*1.E10
 
  !ntot=int8(sum(npartot))


  WRITE(*,*) 'Parameter line read'
  WRITE(*,*) 'Number of files: ', nFiles
  WRITE(*,*) 'Total number of particles in the simulation: ', ntot
  WRITE(*,*) 'Total number of particles in file: ', npart
  WRITE(*,*) 'Masses: ',  mpart
  write(*,*) 'Boxsize',boxsize
  write(*,*) 'Redshift',redshift,'expn',aexpn
  CLOSE(2)




  counts_grid(:,:,:)=0.
   
 
  call omp_set_num_threads(nthreads)

  startall=omp_get_wtime()
  start=omp_get_wtime()   


  ntot_temp=0

  !$OMP PARALLEL DO private(ithread) shared(fbase,counts_grid,nthreads,nfiles)
  DO ithread=0,nthreads-1     
     call cic_dens(fbase,ithread,nthreads,nfiles)
     finish=omp_get_wtime()
     write(*,*)finish-start,'elasped time on thread',ithread
     start=omp_get_wtime()

  end DO
  !$OMP END PARALLEL DO

  finishall=omp_get_wtime()


  write(*,*) finishall-startall,' elapsed time for all files'



  call write_out_cic(fcic,fcount)



 


end program cic

