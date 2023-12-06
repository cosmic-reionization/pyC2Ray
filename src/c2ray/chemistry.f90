module chemistry
    !! Module to compute the time-averaged ionization rates and update electron density

    use, intrinsic :: iso_fortran_env, only: real64

    implicit none

    real(kind=real64), parameter :: epsilon=1e-14_real64                        ! Double precision very small number
    real(kind=real64),parameter :: minimum_fractional_change = 1.0e-3           ! Should be a global parameter. TODO
    real(kind=real64),parameter :: minimum_fraction_of_atoms=1.0e-8
    contains

    subroutine global_pass(dt,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c,conv_flag,m1,m2,m3)
        ! Subroutine Arguments
        real(kind=real64),intent(in) :: dt                          ! time step
        real(kind=real64), intent(in) :: temp(m1,m2,m3)             ! Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)            ! Hydrogen Density Field
        real(kind=real64),intent(inout) :: xh(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64),intent(inout) :: xh_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64),intent(inout) :: xh_intermed(m1,m2,m3)    ! Intermediate ionization fractions of the cells
        real(kind=real64),intent(in) :: phi_ion(m1,m2,m3)           ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64),intent(in) :: bh00                        ! Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64),intent(in) :: albpow                      ! Hydrogen recombination parameter (power law index)
        real(kind=real64),intent(in) :: colh0                       ! Hydrogen collisional ionization parameter
        real(kind=real64),intent(in) :: temph0                      ! Hydrogen ionization energy expressed in K
        real(kind=real64),intent(in) :: abu_c                       ! Carbon abundance
        integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)

        integer,intent(out) :: conv_flag

        integer :: i,j,k  ! mesh position
        ! Mesh position of the cell being treated
        integer,dimension(3) :: pos

        conv_flag = 0
        do k=1,m3
            do j=1,m2
                do i=1,m1
                    pos=(/ i,j,k /)
                    call evolve0D_global(dt,pos,ndens,temp,xh,xh_av,xh_intermed,phi_ion, &
                        bh00,albpow,colh0,temph0,abu_c,conv_flag,m1,m2,m3)
                enddo
            enddo
        enddo

    end subroutine global_pass




    subroutine evolve0D_global(dt,pos,ndens,temp,xh,xh_av,xh_intermed,phi_ion,bh00,albpow,colh0,temph0,abu_c,conv_flag,m1,m2,m3)
        ! Subroutine Arguments
        real(kind=real64),intent(in) :: dt                          ! time step
        integer,dimension(3),intent(in) :: pos                      ! cell position
        real(kind=real64), intent(in) :: temp(m1,m2,m3)             ! Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)            ! Hydrogen Density Field
        real(kind=real64),intent(inout) :: xh(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64),intent(inout) :: xh_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64),intent(inout) :: xh_intermed(m1,m2,m3)    ! Intermediate ionization fractions of the cells
        real(kind=real64),intent(in) :: phi_ion(m1,m2,m3)           ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64),intent(in) :: bh00                        ! Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64),intent(in) :: albpow                      ! Hydrogen recombination parameter (power law index)
        real(kind=real64),intent(in) :: colh0                       ! Hydrogen collisional ionization parameter
        real(kind=real64),intent(in) :: temph0                      ! Hydrogen ionization energy expressed in K
        real(kind=real64),intent(in) :: abu_c                       ! Carbon abundance
        integer,intent(inout) :: conv_flag                          ! convergence counter
        integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)


        ! Local quantities
        real(kind=real64) :: temperature_start
        real(kind=real64) :: ndens_p ! local hydrogen density
        real(kind=real64) :: xh_p ! local ionization fraction
        real(kind=real64) :: xh_av_p ! local mean ionization fraction
        real(kind=real64) :: yh_av_p ! local mean neutral fraction
        real(kind=real64) :: xh_intermed_p ! local mean ionization fraction
        real(kind=real64) :: phi_ion_p ! local ionization rate
        real(kind=real64) :: xh_av_p_old ! mean ion fraction before chemistry (to check convergence)

        ! Initialize local quantities
        temperature_start = temp(pos(1),pos(2),pos(3))
        ndens_p = ndens(pos(1),pos(2),pos(3))
        phi_ion_p = phi_ion(pos(1),pos(2),pos(3))

        ! Initialize local ion fractions
        xh_p = xh(pos(1),pos(2),pos(3))
        xh_av_p = xh_av(pos(1),pos(2),pos(3))
        xh_intermed_p = xh_intermed(pos(1),pos(2),pos(3))
        yh_av_p = 1.0 - xh_av_p
        call do_chemistry(dt,ndens_p,temperature_start,xh_p,xh_av_p,xh_intermed_p,phi_ion_p,bh00,albpow,colh0,temph0,abu_c)

        ! Check for convergence (global flag). In original, convergence is tested using neutral fraction, but testing with
        ! ionized fraction should be equivalent. TODO: add temperature convergence criterion when non-isothermal mode
        ! is added later on.
        xh_av_p_old = xh_av(pos(1),pos(2),pos(3))
        if ((abs(xh_av_p - xh_av_p_old) > minimum_fractional_change .and. &
            abs((xh_av_p - xh_av_p_old) / yh_av_p) > minimum_fractional_change .and. &
            yh_av_p > minimum_fraction_of_atoms) ) then ! Here temperature criterion will be added
            conv_flag = conv_flag + 1
        endif

        ! Put local result in global array
        xh_intermed(pos(1),pos(2),pos(3)) = xh_intermed_p
        xh_av(pos(1),pos(2),pos(3)) = xh_av_p

    end subroutine evolve0D_global
    ! ===============================================================================================
    ! Adapted version of do_chemistry that excludes the "local" part (which is effectively unused in
    ! the current version of c2ray). This subroutine takes grid-arguments along with a position.
    ! Original: G. Mellema (2005)
    ! This version: P. Hirling (2023)
    ! ===============================================================================================
    subroutine do_chemistry(dt,ndens_p,temperature_start,xh_p,xh_av_p,xh_intermed_p,phi_ion_p,bh00,albpow,colh0,temph0,abu_c)
        ! TODO: add clumping argument
        ! Subroutine Arguments
        real(kind=real64),intent(in) :: dt                    ! time step
        real(kind=real64), intent(in) :: temperature_start    ! Local starting temperature
        real(kind=real64), intent(in) :: ndens_p              ! Local Hydrogen Density
        real(kind=real64),intent(inout) :: xh_p               ! HI ionization fractions of the cells
        real(kind=real64),intent(out) :: xh_av_p            ! Time-averaged HI ionization fractions of the cells
        real(kind=real64),intent(out) :: xh_intermed_p       ! Time-averaged HI ionization fractions of the cells
        real(kind=real64),intent(in) :: phi_ion_p             ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64),intent(in) :: bh00                  ! Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64),intent(in) :: albpow                ! Hydrogen recombination parameter (power law index)
        real(kind=real64),intent(in) :: colh0                 ! Hydrogen collisional ionization parameter
        real(kind=real64),intent(in) :: temph0                ! Hydrogen ionization energy expressed in K
        real(kind=real64),intent(in) :: abu_c                 ! Carbon abundance

        real(kind=real64) :: temperature_end, temperature_previous_iteration ! TODO: will be useful when implementing non-isothermal mode
        !real(kind=real64) :: xh0_p                            ! x0 value of the paper. Always used as IC at each iteration (see original do_chemistry)
        real(kind=real64) :: xh_av_p_old                      ! Time-average ionization fraction from previous iteration
        real(kind=real64) :: de                               ! local electron density
        integer :: nit                                        ! Iteration counter

        ! TODO: clumping
        
        ! Initialize IC
        !xh0_p = xh_p
        temperature_end = temperature_start

        nit = 0
        do
            nit = nit + 1
            
            ! Save temperature solution from last iteration
            temperature_previous_iteration = temperature_end

            ! --> Save the values of yh_av found in the previous iteration
            ! --> xh_p_old = xh0_p
            ! -- > xh_av_old = 
            ! -- > yh0_av_old=ion%h_av(0)
            ! -- > yh1_av_old=ion%h_av(1)
            ! At each iteration, the intial condition x(0) is reset. Change happens in the time-average and thus the electron density
            xh_av_p_old = xh_av_p

            ! Calculate (mean) electron density
            ! --> de=electrondens(ndens_p,ion%h_av)
            de = ndens_p * (xh_av_p + abu_c)

            ! --> if (.not.isothermal) call ini_rec_colion_factors(temperature_end%average) 

            ! Calculate the new and mean ionization states
            ! In this version: xh0_p (x0) is used as input, while doric outputs a new x(t) ("xh_av") and <x> ("xh_av_p")
            call doric(xh_p,dt,temperature_end,de,phi_ion_p,bh00,albpow,colh0,temph0,1.0_real64,xh_intermed_p,xh_av_p)
            ! --> de=electrondens(ndens_p,ion%h_av) ---> Why call this a second time ??

            ! --> if (.not.isothermal) &
            ! -->     ! Thermal now takes old values and outputs new values without
            ! -->     ! overwriting the old values .
            ! -->     call thermal(dt, temperature_start%current, &
            ! -->     temperature_end%current, &
            ! -->     temperature_end%average, &
            ! -->     de, ndens_p, ion,phi)    
            
            ! --> Why this local convergence ?
            ! Test for convergence on time-averaged neutral fraction
            ! For low values of this number assume convergence
            if ((abs((xh_av_p-xh_av_p_old)/(1.0_real64 - xh_av_p)) < &
            minimum_fractional_change .or. &
            (1.0_real64 - xh_av_p < minimum_fraction_of_atoms)).and. &
            (abs((temperature_end-temperature_previous_iteration)/ &
            temperature_end) < minimum_fractional_change) & 
            ) then
                exit
            endif

            ! Warn about non-convergence and terminate iteration
            if (nit > 400) then
                ! if (rank == 0) then   
                !     write(logf,*) 'Convergence failing (global) nit=', nit
                !     write(logf,*) 'x',ion%h_av(0)
                !     write(logf,*) 'h',yh0_av_old
                !     write(logf,*) abs(ion%h_av(0)-yh0_av_old)
                ! endif
                ! write(*,*) 'Convergence failing (global) nit=', nit
                !conv_flag = conv_flag + 1 ! TODO: place this at correct location
                exit
            endif
        enddo
    end subroutine do_chemistry



    ! ===============================================================================================
    ! Calculates time dependent ionization state for hydrogen
    ! Author: Garrelt Mellema (2005)
    ! 21 March 2023: adapted for f2py (P. Hirling)
    !
    ! Notes:
    ! Instead of using xfh as size 2 arrays, we only use x (the ionization fraction)
    ! and compute y=1-x whenever necessary. See raytracing_sc.f90 note for more details
    ! This means that everytime that here I write "xh", in original it was "xh(1)"
    ! 
    ! Furthermore, in this version, the subroutine takes "xh_old" (the current ion fraction) as an
    ! argument, and has the updated xh and <xh> as output (in this order).
    ! ===============================================================================================
    subroutine doric (xh_old,dt,temp_p,rhe,phi_p,bh00,albpow,colh0,temph0,clumping,xh,xh_av)

        ! Subroutine Arguments
        real(kind=real64),intent(in) :: xh_old                      ! Current H ionization fraction (t=0), x0 in paper
        real(kind=real64),intent(in) :: dt                          ! time step
        real(kind=real64),intent(in) :: temp_p                      ! local temperature
        real(kind=real64),intent(in) :: rhe                         ! electron density
        real(kind=real64),intent(in) :: phi_p                       ! Local photo-ionization rate
        real(kind=real64),intent(in) :: bh00                        ! Hydrogen recombination parameter (value at 10^4 K)
        real(kind=real64),intent(in) :: albpow                      ! Hydrogen recombination parameter (power law index)
        real(kind=real64),intent(in) :: colh0                       ! Hydrogen collisional ionization parameter
        real(kind=real64),intent(in) :: temph0                      ! Hydrogen ionization energy expressed in K
        real(kind=real64),intent(in) :: clumping                    ! clumping factor

        ! Output
        real(kind=real64),intent(out) :: xh                         ! Updated H ionization fraction
        real(kind=real64),intent(out) :: xh_av                      ! Updated H ionization fraction (time-averaged)

        ! --> TODO: use clumping_module, only: clumping
        ! --> use tped, only: electrondens ! should this really be used inside doric?
        ! --> real(kind=real64),parameter :: sqrtt_isothermal=sqrt(1e4)
        ! --> real(kind=real64),parameter :: acolh0_isothermal=colh0* sqrtt_isothermal*exp(-temph0/1e4)

        real(kind=real64) :: brech0,sqrtt0,acolh0
        ! --> real(kind=real64) :: rhe0
        ! real(kind=real64) :: xh_old
        ! --> real(kind=real64) :: xfh0old
        real(kind=real64) :: aih0
        real(kind=real64) :: delth
        ! --> real(kind=real64) :: eqxfh0
        real(kind=real64) :: eqxh
        real(kind=real64) :: aphoth0
        real(kind=real64) :: deltht,ee
        real(kind=real64) :: avg_factor

        ! find the hydrogen recombination rate at the local temperature
        brech0=clumping*bh00*(temp_p/1e4)**albpow

        ! find the hydrogen collisional ionization rate at the local 
        ! temperature
        sqrtt0=sqrt(temp_p)
        acolh0=colh0*sqrtt0*exp(-temph0/temp_p)

        ! Find the true photo-ionization rate
        aphoth0=phi_p

        ! determine the hydrogen and helium ionization states and 
        ! electron density
        ! (schmidt-voigt & koeppen 1987)
        ! The electron density is the time-averaged value.
        ! This needs to be iterated, in this version the iteration
        ! is assumed to take place outside the doric routine.

        ! Save old values
        !rhe0=rhe
        ! xh_old=xh
        ! --> xfh0old=xh(0)

        aih0=aphoth0+rhe*acolh0
        delth=aih0+rhe*brech0
        eqxh=aih0/delth
        ! --> eqxfh0=rhe*brech0/delth
        deltht=delth*dt
        ee=exp(-deltht)
        xh = (xh_old-eqxh)*ee+eqxh
        ! --> xh(0)=(xfh0old-eqxfh0)*ee+eqxfh0
        !rhe=electrondens(rhh,xfh) ! should this really be used inside doric?

        ! determine neutral densities (take care of precision fluctuations)
        !if (xfh(0) < epsilon .and. abs(xfh(0)).lt.1.0e-10) then
        if (xh < epsilon) then
            xh = epsilon
            ! --> xh(1)=1.0_dp-epsilon
        endif
        ! Determine average ionization fraction over the time step
        ! Mind fp fluctuations. (1.0-ee)/deltht should go to 1.0 for
        ! small deltht, but finite precision leads to values slightly
        ! above 1.0 and for very small values even to 0.0.
        if (deltht.lt.1.0e-8) then
            avg_factor=1.0
        else
            avg_factor=(1.0-ee)/deltht
        endif
        ! The question here is whether it would be better to calculate
        ! xfh_av(0) first, and xfh_av(1) from it.
        xh_av = eqxh+(xh_old-eqxh)*avg_factor
        ! --> xh_av(0)=1.0_dp-xh_av(1)

        ! Take care of precision
        !if (xfh_av(0).lt.epsilon.and.abs(xfh_av(0)).lt.1.0e-10) xfh_av(0)=epsilon
        if (xh_av < epsilon)  xh_av = epsilon

        !rhe=electrondens(rhh,xfh_av) ! more in the spirit of C2-Ray; we could
        ! iterate over rhe inside doric... Add option to doric to do this?
        ! GM/130719
    end subroutine doric
end module chemistry