subroutine cic_dens(fcic,fcicv)
  use globalvars
  implicit none
  character*180::fcic,fcicv
  integer::i
  integer::ix,iy,iz,ixl,ixr,iyl,iyr,izl,izr
  real*8::xr,yr,zr,cellwidth,dx,dy,dz
  real*8::frac_ix,frac_ixl,frac_ixr,frac_iy,frac_iyl,frac_iyr,frac_iz,frac_izl,frac_izr
  real::noned,nonec,kperc

  write(*,*) ncells,fcic

  cellwidth=boxsize/dble(ncells)
  
 

  cic_grid(:,:,:)=0.
  cic_vx(:,:,:)=0.
  cic_vy(:,:,:)=0.
  cic_vz(:,:,:)=0.
  dens_grid(:,:,:)=0.
  write(*,*)'In CIC',boxsize
  
  kperc=0.
  do i=1,ntot
     
     if(mod(i,int(real(ntot)/10.)).eq.0)then
        kperc=kperc+1.
        WRITE(*,*) kperc*10.,'% done',i
!        write(*,*) pos(:,i)
!        write(*,*) vel(:,i)
!        write(*,*) mass(i)
     end if
 




     ix = int((pos(1,i)/boxsize)*real(ncells))+1
     iy = int((pos(2,i)/boxsize)*real(ncells))+1
     iz = int((pos(3,i)/boxsize)*real(ncells))+1

    

     xr=dble(ix)*cellwidth
     yr=dble(iy)*cellwidth
     zr=dble(iz)*cellwidth

     dx=xr-pos(1,i)
     dy=yr-pos(2,i)
     dz=zr-pos(3,i)

     if(dx.lt.0.or.dy.lt.0.or.dz.lt.0) then
        write(*,*) 'dx<0',dx,xr,ix,pos(1,i)
        write(*,*) 'dy<0',dy,yr,iy,pos(2,i)
        write(*,*) 'dz<0',dz,zr,iz,pos(3,i)
        stop
     end if

     if (dx.lt.cellwidth/2.) then
        frac_ix=(cellwidth/2.+dx)/(3.*cellwidth)
        frac_ixr=(cellwidth/2.-dx)/(3.*cellwidth)
        frac_ixl=0.
     else
        frac_ix=((3.*cellwidth)/2.-dx)/(3.*cellwidth)
        frac_ixl=(dx-cellwidth/2.)/(3.*cellwidth)
        frac_ixr=0.
     end if



     if (dy.lt.cellwidth/2.) then
        frac_iy=(cellwidth/2.+dy)/(3.*cellwidth)
        frac_iyr=(cellwidth/2.-dy)/(3.*cellwidth)
        frac_iyl=0.
     else
        frac_iy=((3.*cellwidth)/2.-dy)/(3.*cellwidth)
        frac_iyl=(dy-cellwidth/2.)/(3.*cellwidth)
        frac_iyr=0.
     end if
    
 
     if (dz.lt.cellwidth/2.) then
        frac_iz=(cellwidth/2.+dz)/(3.*cellwidth)
        frac_izr=(cellwidth/2.-dz)/(3.*cellwidth)
        frac_izl=0.
     else
        frac_iz=((3.*cellwidth)/2.-dz)/(3.*cellwidth)
        frac_izl=(dz-cellwidth/2.)/(3.*cellwidth)
        frac_izr=0.
     end if

     if(i.lt.10)then
        write(*,*) i,'_-_-_-'
        write(*,*) ix,iy,iz,ncells,mass(i)
        write(*,*) frac_ix,frac_iy,frac_iz
        write(*,*) frac_ixl,frac_iyl,frac_izl
        write(*,*) frac_ixr,frac_iyr,frac_izr
        write(*,*) frac_ix+frac_ixl+frac_ixr,'ix'
        write(*,*) frac_iy+frac_iyl+frac_iyr,'iy'
        write(*,*) frac_iz+frac_izl+frac_izr,'iz'
        write(*,*) (frac_ix+frac_ixl+frac_ixr+frac_iy+frac_iyl+frac_iyr+frac_iz+frac_izl+frac_izr)

     end if




!     write(*,*) (frac_ix*frac_iy*frac_iz)*mass


     dens_grid(ix,iy,iz)=dens_grid(ix,iy,iz)+mass(i)
     cic_grid(ix,iy,iz)=cic_grid(ix,iy,iz)+frac_ix*frac_iy*frac_iz*mass(i)
     

     cic_vx(ix,iy,iz)=cic_vx(ix,iy,iz)+frac_ix*frac_iy*frac_iz*vel(1,i)*mass(i)
     cic_vy(ix,iy,iz)=cic_vy(ix,iy,iz)+frac_ix*frac_iy*frac_iz*vel(2,i)*mass(i)
     cic_vz(ix,iy,iz)=cic_vz(ix,iy,iz)+frac_ix*frac_iy*frac_iz*vel(3,i)*mass(i)



     ixl=ix-1
     ixr=ix+1
     if (ix+1.gt.ncells) ixr=1
     if (ix-1.lt.1) ixl=ncells

     cic_grid(ixr,iy,iz)=cic_grid(ixr,iy,iz)+frac_ixr*frac_iy*frac_iz*mass(i)
     cic_grid(ixl,iy,iz)=cic_grid(ixl,iy,iz)+frac_ixl*frac_iy*frac_iz*mass(i)

     cic_vx(ixr,iy,iz)=cic_vx(ixr,iy,iz)+frac_ixr*frac_iy*frac_iz*vel(1,i)*mass(i)
     cic_vy(ixr,iy,iz)=cic_vy(ixr,iy,iz)+frac_ixr*frac_iy*frac_iz*vel(2,i)*mass(i)
     cic_vz(ixr,iy,iz)=cic_vz(ixr,iy,iz)+frac_ixr*frac_iy*frac_iz*vel(3,i)*mass(i)
     cic_vx(ixl,iy,iz)=cic_vx(ixl,iy,iz)+frac_ixl*frac_iy*frac_iz*vel(1,i)*mass(i)
     cic_vy(ixl,iy,iz)=cic_vy(ixl,iy,iz)+frac_ixl*frac_iy*frac_iz*vel(2,i)*mass(i)
     cic_vz(ixl,iy,iz)=cic_vz(ixl,iy,iz)+frac_ixl*frac_iy*frac_iz*vel(3,i)*mass(i)


    
     iyl=iy-1
     iyr=iy+1
     if (iy+1.gt.ncells) iyr=1
     if (iy-1.lt.1) iyl=ncells

     cic_grid(ix,iyr,iz)=cic_grid(ix,iyr,iz)+frac_ix*frac_iyr*frac_iz*mass(i)
     cic_grid(ix,iyl,iz)=cic_grid(ix,iyl,iz)+frac_ix*frac_iyl*frac_iz*mass(i)

     cic_vx(ix,iyr,iz)=cic_vx(ix,iyr,iz)+frac_ix*frac_iyr*frac_iz*vel(1,i)*mass(i)
     cic_vy(ix,iyr,iz)=cic_vy(ix,iyr,iz)+frac_ix*frac_iyr*frac_iz*vel(2,i)*mass(i)
     cic_vz(ix,iyr,iz)=cic_vz(ix,iyr,iz)+frac_ix*frac_iyr*frac_iz*vel(3,i)*mass(i)
     cic_vx(ix,iyl,iz)=cic_vx(ix,iyl,iz)+frac_ix*frac_iyl*frac_iz*vel(1,i)*mass(i)
     cic_vy(ix,iyl,iz)=cic_vy(ix,iyl,iz)+frac_ix*frac_iyl*frac_iz*vel(2,i)*mass(i)
     cic_vz(ix,iyl,iz)=cic_vz(ix,iyl,iz)+frac_ix*frac_iyl*frac_iz*vel(3,i)*mass(i)
    
     izl=iz-1
     izr=iz+1
     if (iz+1.gt.ncells) izr=1
     if (iz-1.lt.1) izl=ncells

     cic_grid(ix,iy,izr)=cic_grid(ix,iy,izr)+frac_ix*frac_iy*frac_izr*mass(i)
     cic_grid(ix,iy,izl)=cic_grid(ix,iy,izl)+frac_ix*frac_iy*frac_izl*mass(i)


     cic_vx(ix,iy,izr)=cic_vx(ix,iy,izr)+frac_ix*frac_iy*frac_izr*vel(1,i)*mass(i)
     cic_vy(ix,iy,izr)=cic_vy(ix,iy,izr)+frac_ix*frac_iy*frac_izr*vel(2,i)*mass(i)
     cic_vz(ix,iy,izr)=cic_vz(ix,iy,izr)+frac_ix*frac_iy*frac_izr*vel(3,i)*mass(i)
     cic_vx(ix,iy,izl)=cic_vx(ix,iy,izl)+frac_ix*frac_iy*frac_izl*vel(1,i)*mass(i)
     cic_vy(ix,iy,izl)=cic_vy(ix,iy,izl)+frac_ix*frac_iy*frac_izl*vel(2,i)*mass(i)
     cic_vz(ix,iy,izl)=cic_vz(ix,iy,izl)+frac_ix*frac_iy*frac_izl*vel(3,i)*mass(i)

  end do
  write(*,*) 'done grid',maxval(cic_grid),minval(cic_grid),ncells,sum(cic_grid)
  write(*,*) 'done grid',maxval(dens_grid),minval(dens_grid),ncells,sum(dens_grid)
  write(*,*)sum(mass)
  
  do ix=1,ncells
     do iy=1,ncells
        do iz=i,ncells
           cic_vx(ix,iy,iz)=cic_vx(ix,iy,iz)/cic_grid(ix,iy,iz)
           cic_vy(ix,iy,iz)=cic_vy(ix,iy,iz)/cic_grid(ix,iy,iz)
           cic_vz(ix,iy,iz)=cic_vz(ix,iy,iz)/cic_grid(ix,iy,iz)
        end do
     end do
  end do
  write(*,*) 'writing binary ...'
  open(unit=35,file=fcicv,status='unknown',form='unformatted')
  write(35) ncells,boxsize
  write(35) cic_grid
  write(35) cic_vx
  write(35) cic_vy
  write(35) cic_vz
  write(35) dens_grid
  close(35)
  write(*,*) '... done writing ascii ...'

  open(unit=34,file=fcic,status='unknown')

  nonec=0.
  noned=0.
  do ix=1,ncells
     do iy=1,ncells
        do iz=1,ncells
           !write(*,*) ix,iy,iz,cic_grid(ix,iy,iz)

           if (cic_grid(ix,iy,iz).lt.1)nonec=nonec+1
           if (dens_grid(ix,iy,iz).lt.1)noned=noned+1
           write(34,'(3(i8,1x),e16.8,1x,e16.8)') ix,iy,iz,cic_grid(ix,iy,iz),dens_grid(ix,iy,iz)
!          
        end do
     end do
  end do
  write(*,*) noned/real(ncells**3.),nonec/real(ncells**3.)
  close(34)

  write(*,*) 'done write CIC'
end subroutine cic_dens
