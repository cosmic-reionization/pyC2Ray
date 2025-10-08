module chemistry
    !! Module to compute the time-averaged ionization rates and update electron density

    use, intrinsic :: iso_fortran_env, only: real64

    implicit none

    real(kind=real64), parameter :: epsilon=1e-14_real64                    ! Double precision very small number
    real(kind=real64), parameter :: minimum_fractional_change = 1.0e-3      ! Should be a global parameter. TODO
    real(kind=real64), parameter :: minimum_fraction_of_atoms=1.0e-8
    
    ! cross section constants
    real(kind=real64), parameter :: sigma_H_heth = 1.238e-18                ! HI cross-section at HeI ionization threshold
    real(kind=real64), parameter :: sigma_HeI_at_ion_freq = 7.430e-18       ! HeI cross section at its ionzing frequency 
    real(kind=real64), parameter :: sigma_H_heLya = 9.907e-22               ! HI cross-section at HeII Lya
    real(kind=real64), parameter :: sigma_He_heLya = 1.301e-20              ! HeI cross-section at HeII Lya
    real(kind=real64), parameter :: sigma_H_he2 = 1.230695924714239e-19     ! HI cross-section at HeII ionization threshold
    real(kind=real64), parameter :: sigma_He_he2 = 1.690780687052975e-18    ! HeI cross-section at HeII ionization threshold
    real(kind=real64), parameter :: sigma_HeII_at_ion_freq = 1.589e-18      ! HeII cross section at its ionzing frequency
    
    ! TODO: the variables here below need to be inported by the module rather then being hard-coded
    ! constants for recombination of Heilum
    real(kind=real64), parameter :: p = 0.96_real64      ! Fraction of photons from recombination of HeII that ionize HeI (pag 32 of Kai Yan Lee's thesis)
    real(kind=real64), parameter :: l = 1.425_real64     ! Fraction of photons from 2-photon decay, energetic enough to ionize hydrogen
    real(kind=real64), parameter :: m = 0.737_real64     ! Fraction of photons from 2-photon decay, energetic enough to ionize neutral helium
    real(kind=real64), parameter :: f_lya = 1.0_real64   ! "escape” fraction of Ly α photons, it depends on the neutral fraction
    
    ! cosmological abundance
    real(kind=real64), parameter :: abu_he = 0.074_real64
    real(kind=real64), parameter :: abu_h = 0.926_real64
    real(kind=real64), parameter :: abu_c = 7.1e-7

    ! constants for thermal evolution
    real(kind=real64), parameter :: gamma = 5.0_real64/3.0_real64   ! monoatomic gas heat capacity ratio
    real(kind=real64), parameter :: k_B = 1.380649e-16              ! value from astropy==6.0.0
    real(kind=real64), parameter :: minitemp = 1.0_real64               ! minimum temperature

    contains
    ! TODO: pass the column density to global
    subroutine global_pass(dt,ndens, temp, &
                            xh, xh_av, xh_intermed, &
                            xhei, xhei_av, xhei_intermed, &
                            xheii, xheii_av, xheii_intermed, &
                            phi_hi_ion, phi_hei_ion, phi_heii_ion, &
                            heat_hi_ion, heat_hei_ion, heat_heii_ion, &
                            coldens_hi, coldens_hei, coldens_heii, &
                            clump, conv_flag, m1, m2, m3)
        ! Subroutine Arguments
        real(kind=real64), intent(in) :: dt                         ! time step
        real(kind=real64), intent(in) :: temp(m1,m2,m3)             ! Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)            ! Hydrogen Density Field
        real(kind=real64), intent(inout) :: xh(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xh_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xh_intermed(m1,m2,m3)    ! Intermediate HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei(m1,m2,m3)             ! HeI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei_av(m1,m2,m3)          ! Time-averaged HeI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei_intermed(m1,m2,m3)    ! Intermediate HeI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii(m1,m2,m3)            ! HeII ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii_av(m1,m2,m3)         ! Time-averaged HeII ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii_intermed(m1,m2,m3)   ! Intermediate HeII ionization fractions of the cells
        real(kind=real64), intent(in) :: phi_hi_ion(m1,m2,m3)          ! HI Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: phi_hei_ion(m1,m2,m3)         ! HeI Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: phi_heii_ion(m1,m2,m3)        ! HeII Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: heat_hi_ion(m1,m2,m3)         ! HI Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: heat_hei_ion(m1,m2,m3)        ! HeI Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: heat_heii_ion(m1,m2,m3)       ! HeII Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: coldens_hi(m1,m2,m3)           ! HI column density
        real(kind=real64), intent(in) :: coldens_hei(m1,m2,m3)          ! HeI column density
        real(kind=real64), intent(in) :: coldens_heii(m1,m2,m3)         ! HeII column density
        real(kind=real64), intent(in) :: clump(m1,m2,m3)            ! Clumping factor field (even if it's just a constant it has to be a 3D cube)
        !real(kind=real64), intent(in) :: abu_c                     ! Carbon abundance
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
                    call evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, &
                                        xhei, xhei_av, xhei_intermed, &
                                        xheii, xheii_av, xheii_intermed, &
                                        phi_hi_ion, phi_hei_ion, phi_heii_ion, &
                                        heat_hi_ion, heat_hei_ion, heat_heii_ion, &
                                        coldens_hi, coldens_hei, coldens_heii, &
                                        clump, conv_flag, m1, m2, m3)
                enddo
            enddo
        enddo

    end subroutine global_pass




    subroutine evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, &
                                xhei, xhei_av, xhei_intermed, & 
                                xheii, xheii_av, xheii_intermed, & 
                                phi_hi_ion, phi_hei_ion, phi_heii_ion, &
                                heat_hi_ion, heat_hei_ion, heat_heii_ion, &
                                coldens_hi, coldens_hei, coldens_heii, &
                                clump, conv_flag, m1, m2, m3)
        ! Subroutine Arguments
        real(kind=real64), intent(in) :: dt                         ! time step
        integer,dimension(3),intent(in) :: pos                      ! cell position
        real(kind=real64), intent(in) :: temp(m1,m2,m3)             ! Temperature field
        real(kind=real64), intent(in) :: ndens(m1,m2,m3)            ! Hydrogen Density Field
        real(kind=real64), intent(inout) :: xh(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xh_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xh_intermed(m1,m2,m3)    ! Intermediate ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xhei_intermed(m1,m2,m3)    ! Intermediate ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii(m1,m2,m3)             ! HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii_av(m1,m2,m3)          ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(inout) :: xheii_intermed(m1,m2,m3)    ! Intermediate ionization fractions of the cells
        real(kind=real64), intent(in) :: phi_hi_ion(m1,m2,m3)           ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: phi_hei_ion(m1,m2,m3)          ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: phi_heii_ion(m1,m2,m3)         ! H Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: heat_hi_ion(m1,m2,m3)          ! H Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: heat_hei_ion(m1,m2,m3)         ! H Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: heat_heii_ion(m1,m2,m3)        ! H Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: coldens_hi(m1,m2,m3)           ! HI column density
        real(kind=real64), intent(in) :: coldens_hei(m1,m2,m3)          ! HeI column density
        real(kind=real64), intent(in) :: coldens_heii(m1,m2,m3)         ! HeII column density
        real(kind=real64), intent(in) :: clump(m1,m2,m3)             ! Clumping factor field (even if it's just a constant it has to be a 3D cube)
        !real(kind=real64), intent(in) :: abu_c                      ! Carbon abundance
        integer, intent(inout) :: conv_flag                          ! convergence counter
        integer, intent(in) :: m1                                   ! mesh size x (hidden by f2py)
        integer, intent(in) :: m2                                   ! mesh size y (hidden by f2py)
        integer, intent(in) :: m3                                   ! mesh size z (hidden by f2py)


        ! Local quantities
        real(kind=real64) :: temperature_start
        real(kind=real64) :: ndens_p                        ! local gas density
        real(kind=real64) :: xh_p, xhei_p, xheii_p          ! local hydrogen ionization fraction
        real(kind=real64) :: xh_av_p, xhei_av_p, xheii_av_p ! local hydrogen  mean ionization fraction
        real(kind=real64) :: xh_intermed_p, xhei_intermed_p, xheii_intermed_p! local hydrogen mean ionization fraction
        real(kind=real64) :: yh_av_p        ! local mean neutral fraction TODO: do we still need it? also for He then?
        real(kind=real64) :: phi_hi_ion_p, phi_hei_ion_p, phi_heii_ion_p    ! local photo-ionization rate
        real(kind=real64) :: heat_hi_ion_p, heat_hei_ion_p, heat_heii_ion_p    ! local photo-heating rate
        real(kind=real64) :: coldend_hi_p, coldend_hei_p, coldend_heii_p    ! local photo-heating rate
        real(kind=real64) :: xh_av_p_old, xhei_av_p_old, xheii_av_p_old     ! mean ion fraction before chemistry (to check convergence)
        real(kind=real64) :: clump_p        ! local clumping factor

        ! Initialize local quantities
        temperature_start = temp(pos(1),pos(2),pos(3))
        ndens_p = ndens(pos(1),pos(2),pos(3))
        phi_hi_ion_p = phi_hi_ion(pos(1),pos(2),pos(3))
        phi_hei_ion_p = phi_hei_ion(pos(1),pos(2),pos(3))
        phi_heii_ion_p = phi_heii_ion(pos(1),pos(2),pos(3))
        heat_hi_ion_p = heat_hi_ion(pos(1),pos(2),pos(3))
        heat_hei_ion_p = heat_hei_ion(pos(1),pos(2),pos(3))
        heat_heii_ion_p = heat_heii_ion(pos(1),pos(2),pos(3))
        ! TODO: no need of column density
        coldend_hi_p = coldens_hi(pos(1),pos(2),pos(3))
        coldend_hei_p = coldens_hei(pos(1),pos(2),pos(3))
        coldend_heii_p = coldens_heii(pos(1),pos(2),pos(3))
        clump_p = clump(pos(1),pos(2),pos(3))
        ! TODO: add calculation of the p, y ya2, yb2 and z factor (Table 2 Martina's paper)

        ! Initialize local ion fractions
        xh_p = xh(pos(1),pos(2),pos(3))
        xh_av_p = xh_av(pos(1),pos(2),pos(3))
        xh_intermed_p = xh_intermed(pos(1),pos(2),pos(3))
        xhei_p = xhei(pos(1),pos(2),pos(3))
        xhei_av_p = xhei_av(pos(1),pos(2),pos(3))
        xhei_intermed_p = xhei_intermed(pos(1),pos(2),pos(3))
        xheii_p = xheii(pos(1),pos(2),pos(3))
        xheii_av_p = xheii_av(pos(1),pos(2),pos(3))
        xheii_intermed_p = xheii_intermed(pos(1),pos(2),pos(3))
        !yh_av_p = 1.0 - xh_av_p
        
        call do_chemistry(dt, ndens_p, temperature_start, &
                            xh_p, xh_av_p, xh_intermed_p, &
                            xhei_p, xhei_av_p, xhei_intermed_p, &
                            xheii_p, xheii_av_p, xheii_intermed_p, &
                            phi_hi_ion_p, phi_hei_ion_p, phi_heii_ion_p, &
                            heat_hi_ion_p, heat_hei_ion_p, heat_heii_ion_p, &
                            coldend_hi_p, coldend_hei_p, coldend_heii_p, clump_p)

        ! Check for convergence (global flag). In original, convergence is tested using neutral fraction, but testing with ionized fraction should be equivalent.
        ! TODO: add temperature convergence criterion when non-isothermal mode is added later on.
        xh_av_p_old = xh_av(pos(1),pos(2),pos(3))
        xhe_av_p_old = xhe_av(pos(1),pos(2),pos(3))
        xhei_av_p_old = xhei_av(pos(1),pos(2),pos(3))
        
        ! Hydrogen criterion
        if ((abs(xh_av_p - xh_av_p_old) > minimum_fractional_change .and. &
            abs((xh_av_p - xh_av_p_old) / (1.0 - xh_av_p)) > minimum_fractional_change .and. &
            (1.0 - xh_av_p) > minimum_fraction_of_atoms) ) then
            ! Helium (first ionization) criterion
            if ((abs(xhe_av_p - xhe_av_p_old) > minimum_fractional_change .and. &
                abs((xhe_av_p - xhe_av_p_old) / (1.0 - xhe_av_p)) > minimum_fractional_change .and. &
                (1.0 - xhe_av_p) > minimum_fraction_of_atoms) ) then
                ! Helium (second ionization) criterion
                if ((abs(xhei_av_p - xhei_av_p_old) > minimum_fractional_change .and. &
                    abs((xhei_av_p - xhei_av_p_old) / (1.0 - xhei_av_p)) > minimum_fractional_change .and. &
                    (1.0 - xhei_av_p) > minimum_fraction_of_atoms) ) then 
                    ! TODO: Here temperature criterion will be added
                    conv_flag = conv_flag + 1
                endif
            endif
        endif

        ! Put local result in global array
        xh_intermed(pos(1),pos(2),pos(3)) = xh_intermed_p
        xh_av(pos(1),pos(2),pos(3)) = xh_av_p
        xhe_intermed(pos(1),pos(2),pos(3)) = xhe_intermed_p
        xhe_av(pos(1),pos(2),pos(3)) = xhe_av_p
        xhei_intermed(pos(1),pos(2),pos(3)) = xhei_intermed_p
        xhei_av(pos(1),pos(2),pos(3)) = xhei_av_p

    end subroutine evolve0D_global

    ! ===============================================================================================
    ! Adapted version of do_chemistry that excludes the "local" part (which is effectively unused in
    ! the current version of c2ray). This subroutine takes grid-arguments along with a position.
    ! Original: G. Mellema (2005)
    ! This version: P. Hirling (2023)
    ! ===============================================================================================
    subroutine do_chemistry(dt, ndens_p, temperature_start, & 
                            xh_p, xh_av_p, xh_intermed_p, &
                            xhei_p, xhei_av_p, xhei_intermed_p, &
                            xheii_p, xheii_av_p, xheii_intermed_p, &
                            phi_hi_ion_p, phi_hei_ion_p, phi_heii_ion_p, &
                            heat_hi_ion_p, heat_hei_ion_p, heat_heii_ion_p, &
                            coldhi_p, coldhei_p, coldheii_p, clump_p)
        ! Subroutine Arguments
        real(kind=real64), intent(in) :: dt                    ! time step
        real(kind=real64), intent(in) :: temperature_start    ! Local starting temperature
        real(kind=real64), intent(in) :: ndens_p              ! Local Hydrogen Density
        real(kind=real64), intent(inout) :: xh_p, xhei_p, xheii_p              ! HI ionization fractions of the cells
        real(kind=real64), intent(out) :: xh_av_p, xhei_av_p, xheii_av_p            ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(out) :: xh_intermed_p, xhei_intermed_p, xheii_intermed_p  ! Time-averaged HI ionization fractions of the cells
        real(kind=real64), intent(in) :: phi_hi_ion_p, phi_hei_ion_p, phi_heii_ion_p        ! Photo-ionization rate for the whole grid (called phih_grid in original c2ray)
        real(kind=real64), intent(in) :: heat_hi_ion_p, heat_hei_ion_p, heat_heii_ion_p     ! Photo-heating rate for the whole grid
        real(kind=real64), intent(in) :: coldhi_p, coldhei_p, coldheii_p      ! column density of the three spicies
        real(kind=real64), intent(in) :: clump_p             ! Local clumping factor
        !real(kind=real64), intent(in) :: abu_c                 ! Carbon abundance

        ! Local quantities
        real(kind=real64) :: temperature_end, temperature_previous_iteration ! TODO: will be useful when implementing non-isothermal mode
        real(kind=real64) :: xh_av_p_old, xhei_av_p_old, xheii_av_p_old                      ! Time-average ionization fraction from previous iteration
        real(kind=real64) :: de                               ! local electron density
        integer :: nit                                        ! Iteration counter
        
        ! Initialize IC
        temperature_end = temperature_start

        nit = 0
        do
            nit = nit + 1
            
            ! Save temperature solution from last iteration
            temperature_previous_iteration = temperature_end

            ! At each iteration, the intial condition x(0) is reset. Change happens in the time-average and thus the electron density
            xh_av_p_old = xh_av_p
            xhei_av_p_old = xhei_av_p
            xhei_av_p_old = xheii_av_p

            ! Calculate (mean) electron density
            de = ndens_p * (xh_av_p + xhei_av_p + 2.0d0*xheii_av_p+ abu_c)

            ! TODO: call ini_rec_colion_factors(temperature_end%average) 

            ! Calculate the new and mean ionization states
            ! In this version: xh0_p (x0) is used as input, while doric outputs a new x(t) ("xh_av") and <x> ("xh_av_p")
            ! TODO: multiphase is necessary to correctly calculate the differantial brightness. Hannah's works is on github with helium: https://github.com/garrelt/C2-Ray3Dm1D_Helium/blob/multiphase/code/files_for_3D/evolve_data.F90#L37-L39
            ! TODO: the intermediate need in the python evolve.py for global convergence. Keep it and bring it back.
            call friedrich(dt, temperature_previous_iteration, de, &
                            xh_p, xhei_p, xheii_p, &
                            phi_hi_ion_p, phi_hei_ion_p, phi_heii_ion_p, &
                            heat_hi_ion_p, heat_hei_ion_p, heat_heii_ion_p, &
                            coldhi_p, coldhei_p, coldheii_p, clump_p, &
                            xh_av_p, xhei_av_p, xheii_av_p)
                            !xh_intermed_p, xhei_intermed_p, xheii_intermed_p)

            ! TODO: Call for thermal evolution. It takes the old values and outputs new values without overwriting the old values.
            call thermal(dt, temperature_end, avg_temper, de, ndens_atom, xh_av_p, xhei_av_p, xheii_av_p, heating)
            
            ! Test for convergence on time-averaged neutral fraction. For low values of this number assume convergence
            if ((abs((xh_av_p-xh_av_p_old)/(1.0_real64 - xh_av_p)) < minimum_fractional_change .or. &
                    (1.0_real64 - xh_av_p < minimum_fraction_of_atoms)).and. &
                    (abs((temperature_end-temperature_previous_iteration)/temperature_end) < minimum_fractional_change)) then
                exit
            endif

            ! Warn about non-convergence and terminate iteration
            if (nit > 400) then
                ! TODO: commented out because error message is too verbose
                ! if (rank == 0) then   
                !     write(logf,*) 'Convergence failing (global) nit=', nit
                !     write(logf,*) 'x',ion%h_av(0)
                !     write(logf,*) 'h',yh0_av_old
                !     write(logf,*) abs(ion%h_av(0)-yh0_av_old)
                ! endif
                ! write(*,*) 'Convergence failing (global) nit=', nit
                !conv_flag = conv_flag + 1
                exit
            endif
        enddo
    end subroutine do_chemistry


    ! ===============================================================================================
    ! Calculates time dependent ionization state for hydrogen and helium
    ! Author: Martina Friderich (2012)
    ! 1 November 2024: adapted for f2py (M. Bianco)
    !
    ! Adapted version of Friderich+ (2012) method as an extension to the Altay+ (2008) analytical solution. 
    ! We used Kai Yan Lee PhD thesis as reference. The naming of variables changed a bit to be similar to the variables in the equations described in the thesis.
    ! ===============================================================================================
    subroutine friedrich (dt, temp_p, n_e, &
                            xHII_old, xHeII_old, xHeIII_old, &
                            phi_HI, phi_HeI, phi_HeII, heat_HI, heat_HeI, heat_HeII, &
                            NHI, NHeI, NHeII, clumping, xHII_av, xHeII_av, xHeIII_av)
    
        ! Input & output arguments
        real(kind=real64), intent(in) :: NHI, NHeI, NHeII                   ! column density
        real(kind=real64), intent(in) :: xHII_old, xHeII_old, xHeIII_old    ! previous ionized fractions
        real(kind=real64), intent(in) :: dt, temp_p, n_e                    ! time-step, local temperature and electron number density
        real(kind=real64), intent(in) :: phi_HI, phi_HeI, phi_HeII          ! photo-ionization rates for the three species
        real(kind=real64), intent(in) :: heat_HI, heat_HeI, heat_HeII       ! photo-heating rates for the three species
        real(kind=real64), intent(in) :: clumping                           ! local clumping factor
        real(kind=real64), intent(out) :: xHII_av, xHeII_av, xHeIII_av      ! averaged solution for the ionized fractions

        ! Local variables for Doric methods
        real(kind=real64) :: xHI_av, xHeI_av
        real(kind=real64) :: alphA_HII, alphB_HII, alph1_HII
        real(kind=real64) :: alphA_HeII, alphB_HeII
        real(kind=real64) :: alphA_HeIII, alphB_HeIII, alph1_HeIII, alph2_HeIII
        real(kind=real64) :: nu, sigma_H_heth, sigma_HeI_at_ion_freq, sigma_H_heLya
        real(kind=real64) :: tau_H_heth, tau_He_heth, tau_H_heLya, tau_He_heLya
        real(kind=real64) :: tau_H_he2th, tau_He2_he2th, tau_He_he2th
        real(kind=real64) :: yy, zz, y2a, y2b
        real(kind=real64) :: cHI, cHeI, cHeII, uHI, uHeI, uHeII
        real(kind=real64) :: rHII2HI, rHeII2HI, rHeII2HeI, rHeIII2HI, rHeIII2HeI, rHeIII2HeII
        real(kind=real64) :: S, K, R, T, lamb1, lamb2, lamb3
        real(kind=real64) :: A11, A21, A22, A23, A31, A32, A33, B22, B23
        real(kind=real64) :: c1, c2, c3, p1, p2, p3

        ! Recombination rate of HI (Eq. 2.12 and 2.13)
        alphA_HII = 1.269e-13 * (315608.0d0 / temp_p)**1.503 / (1.0d0 + (604613.0d0 / temp_p)**0.47)**1.923
        alphB_HII = 2.753e-14 * (315608.0d0 / temp_p)**1.5 / (1.0d0 + (115185.0d0 / temp_p)**0.407)**2.242
        alph1_HII = alphA_HII - alphB_HII

        ! Recombination rate of HeII (Eq. 2.14-17)
        if (temp_p < 9.0d3) then
        alphA_HeII = 1.269e-13 * (570662.0d0 / temp_p)**1.503 / (1.0d0 + (1093222.0d0 / temp_p)**0.47)**1.923
        alphB_HeII = 2.753e-14 * (570662.0d0 / temp_p)**1.5 / (1.0d0 + (208271.0d0 / temp_p)**0.407)**2.242
        else
        alphA_HeII = 3.0e-14 * (570662.0d0 / temp_p)**0.654 + 1.9e-3 * temp_p**(-1.5) * exp(-4.7e5 / temp_p) * &
                    (1.0d0 + 0.3 * exp(-9.4e4 / temp_p))
        alphB_HeII = 1.26e-14 * (570662.0d0 / temp_p)**0.75 + 1.9e-3 * temp_p**(-1.5) * exp(-4.7e5 / temp_p) * &
                    (1.0d0 + 0.3 * exp(-9.4e4 / temp_p))
        end if
        
        ! Recombination rate of HeIII (Eq. 2.18-20)
        alphA_HeIII = 2.538e-13 * (1262990.0d0/temp_p)**1.503 / (1.0d0+(2419521.0d0/temp_p)**1.923)**1.923
        alphB_HeIII = 5.506e-14 * (1262990.0d0/temp_p)**1.5 / (1.0d0 + (460945.0d0/temp_p)**0.407)**2.242
        ! Not specified in Kay Yan Lee thesis, but double-checked with Garrelt (13.10.24)
        alph1_HeIII = alphA_HeIII - alphB_HeIII
        alph2_HeIII = 8.54e-11 * temp_p**(-0.6)

        ! two photons emission from recombination of HeIII
        nu = 0.285 * (temp_p/1e4)**0.119

        ! optical depth of HI at HeI ionation frequency threshold
        tau_H_heth  = NHI*sigma_H_heth

        ! optical depth of HeI at HeI ionation frequency threshold
        tau_He_heth = NHeI*sigma_HeI_at_ion_freq 
        
        ! optical depth of H and He at he+Lya (40.817eV)
        tau_H_heLya = NHI*sigma_H_heLya
        tau_He_heLya= NHeI*sigma_He_heLya
        
        ! optical depth of H at HeII ion threshold
        tau_H_he2th = NHI*sigma_H_he2
        
        ! optical depth of HeI at HeII ion threshold
        tau_He_he2th = NHeI*sigma_He_he2
        
        ! optical depth of HeII at HeII ion threshold
        tau_He2_he2th = NHeII*sigma_HeII_at_ion_freq
        
        ! Ratios of these optical depths needed in doric
        yy = tau_H_heth /(tau_H_heth +tau_He_heth)
        zz = tau_H_heLya/(tau_H_heLya+tau_He_heLya)
        y2a =  tau_He2_he2th /(tau_He2_he2th +tau_He_he2th+tau_H_he2th)
        y2b =  tau_He_he2th /(tau_He2_he2th +tau_He_he2th+tau_H_he2th)

        ! Collisional ionization process (Eq. 2.21-23)
        ! TODO: a remarks is that in principle collisional ionization is also clumping dependent (but HI clumping) but probably irrelevant at this scale.
        cHI = 5.835e-11 * sqrt(temp_p) * exp(-157804.0d0/temp_p)
        cHeI = 2.71e-11 * sqrt(temp_p) * exp(-285331.0d0/temp_p)
        cHeII = 5.707e-12 * sqrt(temp_p) * exp(-631495.0d0/temp_p)

        ! Photo-ionization rates (Eq. 2.27-29)
        uHI = phi_HI + cHI * n_e
        uHeI = phi_HeI + cHeI * n_e
        uHeII = phi_HeII + cHeII * n_e

        ! Recombination rate (Eq. 2.30-35)
        rHII2HI = -alphB_HII
        rHeII2HI = p*alphA_HeII + yy*alph1_HeIII
        rHeII2HeI = (1-yy)*alph1_HII - alphA_HeII
        rHeIII2HI = (1-y2a-y2b)*alph1_HeIII + alph2_HeIII + (nu*(l-m+m*yy)+(1-nu)*f_lya*zz)*alphB_HeIII
        rHeIII2HeI = y2b*alph1_HeIII + (nu*m*(1-yy)+(1-nu)*f_lya*(1-zz))*alphB_HeIII + alphA_HeIII - y2a*alph1_HeIII
        rHeIII2HeII = y2a*alph1_HeIII - alphA_HeIII

        ! get matrix elements
        A11 = -uHI + rHII2HI
        !A12 = 0.
        !A13 = 0.
        A21 = abu_he/abu_h * rHeII2HI * n_e
        A22 = -uHeI - uHeII + rHeII2HeI * n_e
        A23 = uHeII
        A31 = abu_he/abu_h * rHeIII2HI * n_e
        A32 = -uHeI + rHeIII2HeI * n_e
        A33 = rHeIII2HeII * n_e

        ! define coefficients 
        S = sqrt(A33**2.0d0 - 2.0d0*A33*A22 + A22**2.0d0 + 4.0d0*A32*23d0)
        K = 1.0d0/(A23*A32 - A33*A22)
        R = 2.0d0*A23*(A33*uHI*K - xHeII_old)
        T = -A32*uHeI*K - xHeIII_old

        ! define eigen-value
        lamb1  = A11
        lamb2 = 0.5d0*(A33 + A22 - S)
        lamb3 = 0.5d0*(A33 + A22 + S)

        !p1 = -(uHI + (A33*A12 - A32*A13)*uHeI*K) / A11
        p1 = -uHI/A11
        p2 = A33*uHeI*K
        p3 = -A32*uHeI*K

        !B11 = 1.0
        !B12 = (-2.0*A32*A13 + A12 *(A33-A22+S)) / (2.0*A32*(A11-lamb2))
        !B12 = 0.0
        !B13 = (-2.0*A32*A13 + A12 *(A33-A22-S)) / (2.0*A32*(A11-lamb3))
        !B13 = 0.0
        !B21 = 0.0
        B22 = (-A33+A22-S) / (2.0d0*A32)
        B23 = (-A33+A22+S) / (2.0d0*A32)
        !B31 = 0.0
        !B32 = 1.0
        !B33 = 1.0

        c1 = (2.0d0*p1*S - (R+(A33-A22)*T)*(A21 - A31)) / 2.0d0*S + xHII_old + T/2.0d0*(A21+A31)
        c2 = (R + (A33 - A22 - S)*T) / (2.0d0*S)
        c3 = -(R + (A33 - A22 + S)*T) / (2.0d0*S)

        !xHII_av = B11*c1/(lamb1*dt)*(exp(lamb1*dt)-1.0)+B12*c2/(lamb2*dt)(exp(lamb2*dt)-1.0) + B13*c3/(lamb3*dt)*(np.exp(lamb3*dt)-1.0)
        xHII_av = c1/(lamb1*dt)*(exp(lamb1*dt)-1.0)
        xHI_av = 1.0 - xHII_av
        !xHeII_av = B21*c1/(lamb1*dt)*(exp(lamb1*dt)-1.0)+B22*c2/(lamb2*dt)*(exp(lamb2*dt)-1.0) + B23*c3/(lamb3*dt)*(exp(lamb3*dt)-1.0)
        xHeII_av = B22*c2/(lamb2*dt)*(exp(lamb2*dt)-1.0) + B23*c3/(lamb3*dt)*(exp(lamb3*dt)-1.0)
        xHeIII_av = c2/(lamb2*dt)*(exp(lamb2*dt)-1.0) + c3/(lamb3*dt)*(exp(lamb3*dt)-1.0)
        xHeI_av = 1.0 - xHeII_av - xHeIII_av

    end subroutine friedrich

    ! TODO: here after there should be the heating part (from eq 2.69 in Kay Lee thesis, pag 37)
    subroutine thermal(dt, end_temper, avg_temper, ndens_electron, ndens_atom, xhi_p, xhei_p, xheii_p, heating)
    
        ! The time step
        real(kind=real64), intent(in) :: dt
        ! end time temperature of the cell
        real(kind=real64), intent(inout) :: end_temper
        ! average temperature of the cell
        real(kind=real64), intent(out) :: avg_temper
        ! Electron density of the cell
        real(kind=real64), intent(in) :: ndens_electron
        ! Number density of atoms of the cell
        real(kind=real64), intent(in) :: ndens_atom
        ! Photo-heating rate of the cells
        real(kind=real64) intent(in) :: heating
        ! Ionized fraction of the cell
        type(ionstates), intent(in) :: xhi_p, xhei_p, xheii_p

        ! initial temperature
        real(kind=real64) :: initial_temp
        ! timestep taken to solve the ODE
        real(kind=real64) :: dt_ODE
        ! timestep related to thermal timescale
        real(kind=real64) :: dt_thermal
        ! record the time elapsed
        real(kind=real64) :: cumulative_time
        ! internal energy of the cell
        real(kind=real64) :: internal_energy
        ! thermal timescale, used to calculate the thermal timestep
        real(kind=real64) :: thermal_timescale
        ! cooling rate
        real(kind=real64) :: cooling
        ! difference of heating and cooling rate
        real(kind=real64) :: thermal_rate
        ! cosmological cooling rate
        real(kind=real64) :: cosmo_cool_rate
        ! Counter of number of thermal timesteps taken
        integer :: i_heating

        real(kind=real64) :: pressr !< pressure
        internal_energy = (ndens+ndens_electron)*k_B*end_temper/(1.0d0-gamma)

        ! Thermal process is only done if the temperature of the cell is larger than the minimum temperature requirement
        if (end_temper > minitemp) then

            ! stores the time elapsed is done
            cumulative_time = 0.0 
        
            ! initialize the counter
            i_heating = 0

            ! initialize time averaged temperature
            avg_temper = 0.0 

            ! initial temperature
            initial_temp = end_temper

            ! thermal process begins
            do
                ! update heating counter TODO: don't know if necessary but needed maybe for the table
                i_heating = i_heating+1

                ! update cooling rate from cooling tables
                ! TODO: read the cooling tables (?) see function in cooling.f90
                !cooling 

                ! Find total energy change rate
                thermal_rate = max(1d-50, abs(cooling-heating))
                ! TODO: continue checking here after

                ! Calculate thermal time scale
                thermal_timescale = internal_energy/abs(thermal_rate)

                ! Calculate time step needed to limit energy change to a fraction relative_denergy
                dt_thermal = relative_denergy*thermal_timescale

                ! Time step to large, change it to dt_thermal. 
                ! Make sure we do not integrate for longer than the total time step
                dt_ODE = min(dt_thermal,dt-cumulative_time)

                ! Find new internal energy density
                internal_energy = internal_energy+dt_ODE*(heating-cooling)

                ! Update avg_temper sum (first part of dt_thermal sub time step)
                avg_temper = avg_temper+0.5*end_temper*dt_ODE

                ! Find new temperature from the internal energy density
                end_temper = pressr2temper(internal_energy*gamma1,ndens_atom, &
                    electrondens(ndens_atom,ion%h_av,ion%he_av))

                ! Update avg_temper sum (second part of dt_thermal sub time step)
                avg_temper = avg_temper+0.5*end_temper*dt_ODE
                            
                ! Take measures if temperature drops below minitemp
                if (end_temper < minitemp) then
                    internal_energy = temper2pressr(minitemp,ndens_atom, &
                                        electrondens(ndens_atom,ion%h_av,ion%he_av))
                    end_temper = minitemp
                endif
                            
                ! Update fractional cumulative_time
                cumulative_time = cumulative_time+dt_ODE

                ! Exit if we reach dt
                if (cumulative_time >= dt.or.abs(cumulative_time-dt) < 1e-6*dt) exit

                ! In case we spend too much time here, we exit
                if (i_heating > 10000) exit
            enddo

        endif

    end subroutine thermal

end module chemistry