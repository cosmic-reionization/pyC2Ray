subroutine write_out_cic(fcic,fcount,ferr)
  use globalvars
  implicit none
  character*180::fcic,fcount,ferr
  integer::ix,iy,iz,slab
  real*8::npcic,npcount
  integer*8::nempty_cells,nempty_cells_cic
  real,allocatable::cic_grid_real(:,:),cic_vx_real(:,:),cic_vy_real(:,:),cic_vz_real(:,:)
  real,allocatable::count_grid_real(:,:),count_vx_real(:,:),count_vy_real(:,:),count_vz_real(:,:)
  real::real_boxsize,real_mpart
  integer :: i, j


  open(unit=15,file=ferr,status='unknown')

  write(15,*) 'finalizing CIC'
  write(*,*) 'finalizing CIC',ferr


  nempty_cells_cic=0
  nempty_cells=0
  npcic=0
  npcount=0
  do ix=1,ncells
     do iy=1,ncells
        do iz=1,ncells
        
           npcount=npcount+count_grid(ix,iy,iz) ! this is wrong since its an integer sum and cic_grid is dble
           npcic=npcic+cic_grid(ix,iy,iz)
        
           if (count_grid(ix,iy,iz).eq.0)nempty_cells=nempty_cells+1
           if (cic_grid(ix,iy,iz).eq.0)nempty_cells_cic=nempty_cells_cic+1

           cic_vx(ix,iy,iz)=cic_vx(ix,iy,iz)/cic_grid(ix,iy,iz)
           cic_vy(ix,iy,iz)=cic_vy(ix,iy,iz)/cic_grid(ix,iy,iz)
           cic_vz(ix,iy,iz)=cic_vz(ix,iy,iz)/cic_grid(ix,iy,iz)

           count_vx(ix,iy,iz)=count_vx(ix,iy,iz)/count_grid(ix,iy,iz)
           count_vy(ix,iy,iz)=count_vy(ix,iy,iz)/count_grid(ix,iy,iz)
           count_vz(ix,iy,iz)=count_vz(ix,iy,iz)/count_grid(ix,iy,iz)
      

        end do
     end do
  end do

  write(15,*) 'N empty cells (CIC)   : ', nempty_cells_cic
  write(15,*) '              (COUNTS): ', nempty_cells

  
  write(15,*) '    : CIC'
  write(15,*) '    : vx',maxval(cic_vx),minval(cic_vx),maxloc(cic_vx)
  write(15,*) '    : vy',maxval(cic_vy),minval(cic_vy),maxloc(cic_vy)
  write(15,*) '    : vz',maxval(cic_vz),minval(cic_vz),maxloc(cic_vz)

  write(15,*) '    : COUNT'
  write(15,*) '    : vx',maxval(count_vx),minval(count_vx),maxloc(count_vx)
  write(15,*) '    : vy',maxval(count_vy),minval(count_vy),maxloc(count_vy)
  write(15,*) '    : vz',maxval(count_vz),minval(count_vz),maxloc(count_vz)

 





  real_boxsize=real(boxsize)
  real_mpart=real(mpart)


  write(15,*) 'writing binary ... to',fcic
  open(unit=35,file=fcic,status='unknown',form='unformatted')

  write(35) ncells
  write(35) ntot
  write(35) real_boxsize
  write(35) real_mpart

  write(15,*) 'ncells: ', ncells   
  write(15,*) 'ntot: ', ntot
  write(15,*) 'real_boxsize: ', real_boxsize
  write(15,*) 'real_mpart: ', real_mpart
  !do slab=1,ncells
   !  if (allocated(cic_grid_real)) deallocate(cic_grid_real)
    ! allocate(cic_grid_real(ncells,ncells))

     !cic_grid_real(:,:)=real(cic_grid(:,:,slab))

     !write(35) cic_grid_real
     
  !end do
   write(35) cic_grid
   !print*, '  Minimum value of cic_grid_real: ', minval(cic_grid)
   !print*, '  Maximum value of cic_grid_real: ', maxval(cic_grid)
   !print*, 'First value: ', cic_grid(1,1,1)
   !print*, 'cic_grid(2,1,1): ', cic_grid(2,1,1)
   !print*, 'cic_grid(1,2,1) ', cic_grid(1,2,1)
   !print*, 'cic_grid(256,1,1) ', cic_grid(256,1,1)
   !print*, 'Last value: ', cic_grid(ncells,ncells,ncells)
  
  !deallocate(cic_grid,cic_grid_real)
   deallocate(cic_grid)


  do slab=1,ncells
     if (allocated(cic_vx_real)) deallocate(cic_vx_real)
     allocate(cic_vx_real(ncells,ncells))

     cic_vx_real(:,:)=real(cic_vx(:,:,slab))
     write(35) cic_vx_real
  end do
  deallocate(cic_vx,cic_vx_real)


  do slab=1,ncells
     if (allocated(cic_vy_real)) deallocate(cic_vy_real)
     allocate(cic_vy_real(ncells,ncells))

     cic_vy_real(:,:)=real(cic_vy(:,:,slab))
     write(35) cic_vy_real
  end do
  deallocate(cic_vy,cic_vy_real)


  do slab=1,ncells
     if (allocated(cic_vz_real)) deallocate(cic_vz_real)
     allocate(cic_vz_real(ncells,ncells))

     cic_vz_real(:,:)=real(cic_vz(:,:,slab))
     write(35) cic_vz_real
  end do
  deallocate(cic_vz,cic_vz_real)


  close(35)





  write(15,*) 'writing binary ... to',fcount
  open(unit=35,file=fcount,status='unknown',form='unformatted')

  write(35) ncells
  write(35) ntot
  write(35) real_boxsize
  write(35) real_mpart
  do slab=1,ncells
     if (allocated(count_grid_real)) deallocate(count_grid_real)
     allocate(count_grid_real(ncells,ncells))

     count_grid_real(:,:)=real(count_grid(:,:,slab))
     write(35) count_grid_real
  end do
  deallocate(count_grid,count_grid_real)



  do slab=1,ncells
     if (allocated(count_vx_real)) deallocate(count_vx_real)
     allocate(count_vx_real(ncells,ncells))

     count_vx_real(:,:)=real(count_vx(:,:,slab))
     write(35) count_vx_real
  end do
  deallocate(count_vx,count_vx_real)


  do slab=1,ncells
     if (allocated(count_vy_real)) deallocate(count_vy_real)
     allocate(count_vy_real(ncells,ncells))

     count_vy_real(:,:)=real(count_vy(:,:,slab))
     write(35) count_vy_real
  end do
  deallocate(count_vy,count_vy_real)


  do slab=1,ncells
     if (allocated(count_vz_real)) deallocate(count_vz_real)
     allocate(count_vz_real(ncells,ncells))

     count_vz_real(:,:)=real(count_vz(:,:,slab))
     write(35) count_vz_real
  end do
  deallocate(count_vz,count_vz_real)



  close(45)



  write(15,*) '... done writing ...',ncells
  write(15,*) 'done write COUNT of',npcount,'particles',ntot
  write(15,*) 'done write CIC of',npcic,'particles',ntot

  write(*,*) '... done writing ...',ncells
  write(*,*) 'done write COUNT of',npcount,'particles',ntot !need to be compared with total mass not ntot
  write(*,*) 'done write CIC of',npcic,'particles',ntot

end subroutine write_out_cic
