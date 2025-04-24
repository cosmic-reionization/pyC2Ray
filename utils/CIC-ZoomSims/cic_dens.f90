subroutine cic_dens(fbase,ithread,nthread,nfiles)
  use globalvars
  implicit none
  integer*8::i,j,iunit
  integer::ix,iy,iz,ix1,iy1,iz1,ifile,ithread,nthread
  real*8::x0,y0,z0,x1,y1,z1,cellwidth,dx0,dy0,dz0,dxs,dys,dzs
  real*8::nx0y0z0,nxsy0z0,nx0ysz0,nx0y0zs,nxsysz0,nxsy0zs,nx0yszs,nxsyszs
  character*180::part_file,fbase
  character*4::nfile_str
  real*8,allocatable::cic_loc_grid(:,:,:),count_loc_grid(:,:,:),cic_loc_mass(:,:,:),count_loc_mass(:,:,:)
  real*8,allocatable::cic_loc_vx(:,:,:),cic_loc_vy(:,:,:),cic_loc_vz(:,:,:)
  real*8,allocatable::count_loc_vx(:,:,:),count_loc_vy(:,:,:),count_loc_vz(:,:,:)

  real,allocatable::pos(:,:),vel(:,:),mass(:)
  integer::npart_in_file,Npart_all_types
 
  CHARACTER*20 :: conv='LITTLE_ENDIAN'
  INTEGER, DIMENSION(6) :: npart,npartot
  DOUBLE PRECISION :: h,Lambda,Om,aexpn,redshift,box_dp
  INTEGER :: n_o_b,SF,Fb,nFiles,Cool
  DOUBLE PRECISION, DIMENSION(6) :: masses
  CHARACTER*60 :: c5
  CHARACTER*4 :: header

 



  allocate(cic_loc_grid(ncells,ncells,ncells),cic_loc_vx(ncells,ncells,ncells),cic_loc_vy(ncells,ncells,ncells),cic_loc_vz(ncells,ncells,ncells))
  allocate(count_loc_grid(ncells,ncells,ncells),count_loc_vx(ncells,ncells,ncells),count_loc_vy(ncells,ncells,ncells),count_loc_vz(ncells,ncells,ncells))
  allocate(cic_loc_mass(ncells,ncells,ncells),count_loc_mass(ncells,ncells,ncells))
  
  
  cic_loc_mass(:,:,:)=0.
  count_loc_mass(:,:,:)=0.

  cic_loc_grid(:,:,:)=0.
  cic_loc_vx(:,:,:)=0.
  cic_loc_vy(:,:,:)=0.
  cic_loc_vz(:,:,:)=0.


  count_loc_grid(:,:,:)=0.
  count_loc_vx(:,:,:)=0.
  count_loc_vy(:,:,:)=0.
  count_loc_vz(:,:,:)=0.

  do j=0,(nfiles/nthread)-1
     

     ifile=ithread*nfiles/nthread+j
     
     
     
     WRITE(nfile_str,'(i4)') ifile
     nfile_str=ADJUSTL(nfile_str)
     
     
     part_file=trim(adjustl(fbase))//trim(adjustl(nfile_str))
     write(*,*) 'thread',ithread,' reading file',part_file     
     
  




     iunit=ithread+21

     OPEN(iunit, FILE=part_file,FORM='unformatted',STATUS='OLD',CONVERT=conv)
!     READ(iunit)n_o_b
     
     ! Reading the parameters line
     
     READ(iunit) npart(1:6),masses(1:6),aexpn,redshift,SF,Fb,npartot(1:6),Cool, &
          & nFiles,box_dp,Om,Lambda,h,c5
     
     
     ! Calculating masses and total number of particles
     
     Npart_all_types=0
     DO i = 1,6
        Npart_all_types=Npart_all_types+npart(i)
     ENDDO
     
     WRITE(*,*) 'Npart_all_types - ',Npart_all_types
     npart_in_file=Npart_all_types

     if (allocated(pos)) deallocate(pos,vel,mass)
     allocate(pos(3,npart_in_file),vel(3,npart_in_file),mass(npart_in_file))


     
!     write(*,*) header,'H',n_o_b
     READ(iunit) pos
     
     !PRINT *,' Positions read from file ',ifile,pos(:,1)
     ! Velocities
     READ(iunit) vel
     READ(iunit)  ! dont read ids - we dont need them
     READ(iunit) mass

     pos=pos/1000.
     close(iunit)

     
     


     cellwidth=boxsize/dble(ncells)
     
     do i=1,npart_in_file
        
            
        
        ix = int((pos(1,i)/boxsize)*real(ncells))+1
        iy = int((pos(2,i)/boxsize)*real(ncells))+1
        iz = int((pos(3,i)/boxsize)*real(ncells))+1
      
        if (ix .lt. 1) ix=ncells
        if (iy .lt. 1) iy=ncells
        if (iz .lt. 1) iz=ncells
        if (ix .gt. ncells) ix=1
        if (iy .gt. ncells) iy=1
        if (iz .gt. ncells) iz=1
        
        
        
        
        
        x0=dble(ix)*cellwidth-cellwidth/2.
        y0=dble(iy)*cellwidth-cellwidth/2.
        z0=dble(iz)*cellwidth-cellwidth/2.
        
        
        if (pos(1,i) .gt. x0) ix1=ix+1
        if (pos(2,i) .gt. y0) iy1=iy+1
        if (pos(3,i) .gt. z0) iz1=iz+1
        
        
        
        if (pos(1,i) .lt. x0) ix1=ix-1
        if (pos(2,i) .lt. y0) iy1=iy-1
        if (pos(3,i) .lt. z0) iz1=iz-1
        
        
        
        x1=dble(ix1)*cellwidth-cellwidth/2.
        y1=dble(iy1)*cellwidth-cellwidth/2.
        z1=dble(iz1)*cellwidth-cellwidth/2.
     

        
    
        !THIS IS CORRECTED
        dx0=abs(x1-pos(1,i))
        dy0=abs(y1-pos(2,i))
        dz0=abs(z1-pos(3,i))
        dxs=abs(pos(1,i)-x0)
        dys=abs(pos(2,i)-y0)
        dzs=abs(pos(3,i)-z0)




        nx0y0z0=(1./cellwidth**3.)*dx0*dy0*dz0
        
        nxsy0z0=(1./cellwidth**3.)*dxs*dy0*dz0
        nx0ysz0=(1./cellwidth**3.)*dx0*dys*dz0
        nx0y0zs=(1./cellwidth**3.)*dx0*dy0*dzs
        
        nxsysz0=(1./cellwidth**3.)*dxs*dys*dz0
        nxsy0zs=(1./cellwidth**3.)*dxs*dy0*dzs
        nx0yszs=(1./cellwidth**3.)*dx0*dys*dzs
        
        nxsyszs=(1./cellwidth**3.)*dxs*dys*dzs
        

        if (ix1 .gt.ncells) ix1=1
        if (iy1 .gt.ncells) iy1=1
        if (iz1 .gt.ncells) iz1=1
        if (ix1 .lt.1) ix1=ncells
        if (iy1 .lt.1) iy1=ncells
        if (iz1 .lt.1) iz1=ncells
        
!!$        
        count_loc_grid(ix,iy,iz)=count_loc_grid(ix,iy,iz)+dble(mass(i))
        count_loc_vx(ix,iy,iz)=count_loc_vx(ix,iy,iz)+dble(vel(1,i))
        count_loc_vy(ix,iy,iz)=count_loc_vy(ix,iy,iz)+dble(vel(2,i))
        count_loc_vz(ix,iy,iz)=count_loc_vz(ix,iy,iz)+dble(vel(3,i))
        
        
        

        cic_loc_grid(ix,iy,iz)=cic_loc_grid(ix,iy,iz)+nx0y0z0*dble(mass(i))

        cic_loc_grid(ix1,iy,iz)=cic_loc_grid(ix1,iy,iz)+nxsy0z0*dble(mass(i))
        cic_loc_grid(ix,iy1,iz)=cic_loc_grid(ix,iy1,iz)+nx0ysz0*dble(mass(i))
        cic_loc_grid(ix,iy,iz1)=cic_loc_grid(ix,iy,iz1)+nx0y0zs*dble(mass(i))
        cic_loc_grid(ix1,iy1,iz)=cic_loc_grid(ix1,iy1,iz)+nxsysz0*dble(mass(i))
        cic_loc_grid(ix1,iy,iz1)=cic_loc_grid(ix1,iy,iz1)+nxsy0zs*dble(mass(i))
        cic_loc_grid(ix,iy1,iz1)=cic_loc_grid(ix,iy1,iz1)+nx0yszs*dble(mass(i))
        cic_loc_grid(ix1,iy1,iz1)=cic_loc_grid(ix1,iy1,iz1)+nxsyszs*dble(mass(i))
        
!!$        
        cic_loc_vx(ix,iy,iz)=cic_loc_vx(ix,iy,iz)+nx0y0z0*dble(vel(1,i))
        cic_loc_vx(ix1,iy,iz)=cic_loc_vx(ix1,iy,iz)+nxsy0z0*dble(vel(1,i))
        cic_loc_vx(ix,iy1,iz)=cic_loc_vx(ix,iy1,iz)+nx0ysz0*dble(vel(1,i))
        cic_loc_vx(ix,iy,iz1)=cic_loc_vx(ix,iy,iz1)+nx0y0zs*dble(vel(1,i))
        cic_loc_vx(ix1,iy1,iz)=cic_loc_vx(ix1,iy1,iz)+nxsysz0*dble(vel(1,i))
        cic_loc_vx(ix1,iy,iz1)=cic_loc_vx(ix1,iy,iz1)+nxsy0zs*dble(vel(1,i))
        cic_loc_vx(ix,iy1,iz1)=cic_loc_vx(ix,iy1,iz1)+nx0yszs*dble(vel(1,i))
        cic_loc_vx(ix1,iy1,iz1)=cic_loc_vx(ix1,iy1,iz1)+nxsyszs*dble(vel(1,i))
        
        
        cic_loc_vy(ix,iy,iz)=cic_loc_vy(ix,iy,iz)+nx0y0z0*dble(vel(2,i))
        cic_loc_vy(ix1,iy,iz)=cic_loc_vy(ix1,iy,iz)+nxsy0z0*dble(vel(2,i))
        cic_loc_vy(ix,iy1,iz)=cic_loc_vy(ix,iy1,iz)+nx0ysz0*dble(vel(2,i))
        cic_loc_vy(ix,iy,iz1)=cic_loc_vy(ix,iy,iz1)+nx0y0zs*dble(vel(2,i))
        cic_loc_vy(ix1,iy1,iz)=cic_loc_vy(ix1,iy1,iz)+nxsysz0*dble(vel(2,i))
        cic_loc_vy(ix1,iy,iz1)=cic_loc_vy(ix1,iy,iz1)+nxsy0zs*dble(vel(2,i))
        cic_loc_vy(ix,iy1,iz1)=cic_loc_vy(ix,iy1,iz1)+nx0yszs*dble(vel(2,i))
        cic_loc_vy(ix1,iy1,iz1)=cic_loc_vy(ix1,iy1,iz1)+nxsyszs*dble(vel(2,i))
        
        
        cic_loc_vz(ix,iy,iz)=cic_loc_vz(ix,iy,iz)+nx0y0z0*dble(vel(3,i))
        cic_loc_vz(ix1,iy,iz)=cic_loc_vz(ix1,iy,iz)+nxsy0z0*dble(vel(3,i))
        cic_loc_vz(ix,iy1,iz)=cic_loc_vz(ix,iy1,iz)+nx0ysz0*dble(vel(3,i))
        cic_loc_vz(ix,iy,iz1)=cic_loc_vz(ix,iy,iz1)+nx0y0zs*dble(vel(3,i))
        cic_loc_vz(ix1,iy1,iz)=cic_loc_vz(ix1,iy1,iz)+nxsysz0*dble(vel(3,i))
        cic_loc_vz(ix1,iy,iz1)=cic_loc_vz(ix1,iy,iz1)+nxsy0zs*dble(vel(3,i))
        cic_loc_vz(ix,iy1,iz1)=cic_loc_vz(ix,iy1,iz1)+nx0yszs*dble(vel(3,i))
        cic_loc_vz(ix1,iy1,iz1)=cic_loc_vz(ix1,iy1,iz1)+nxsyszs*dble(vel(3,i))

     end do

     deallocate(pos,vel,mass)
     
  end do
 

  !$OMP CRITICAL
  cic_vx=cic_vx+cic_loc_vx
  cic_vy=cic_vy+cic_loc_vy
  cic_vz=cic_vz+cic_loc_vz

  cic_grid=cic_grid+cic_loc_grid

  count_vx=count_vx+count_loc_vx
  count_vy=count_vy+count_loc_vy
  count_vz=count_vz+count_loc_vz
  count_grid=count_grid+count_loc_grid
 
  !$OMP END CRITICAL

  
  deallocate(cic_loc_grid,cic_loc_vx,cic_loc_vy,cic_loc_vz)
  deallocate(count_loc_grid,count_loc_vx,count_loc_vy,count_loc_vz)

end subroutine cic_dens
