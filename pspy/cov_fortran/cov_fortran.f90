! FFLAGS="-fopenmp -fPIC -Ofast -ffree-line-length-none" f2py-2.7 -c -m cov_fortran cov_fortran.f90 wigner3j_sub.f -lgomp

subroutine calc_cov_spin0_single_win(wcl,cov_array)
    implicit none
    real(8), intent(in)    :: wcl(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (wcl(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine


subroutine calc_cov_spin0_single_win_threshold(wcl, threshold, cov_array)
    implicit none
    real(8), intent(in)    :: wcl(:)
    integer, intent(in)   :: threshold
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, min(l1+threshold,nlmax)

            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (wcl(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine

subroutine calc_cov_spin0(ac_bd,ad_bc,cov_array)
    implicit none
    real(8), intent(in)    :: ac_bd(:),ad_bc(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2), fac
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,fac,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i   = l3-lmin+1
                cov_array(l1-1,l2-1,1) =cov_array(l1-1,l2-1,1)+ (ac_bd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1,l2-1,2) =cov_array(l1-1,l2-1,2)+ (ad_bc(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do
end subroutine



subroutine calc_cov_spin0and2_single_win_simple(wcl,cov_array)
    implicit none
    real(8), intent(in)    :: wcl(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2)
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i = l3-lmin+1
                cov_array(l1-1,l2-1, 1) = cov_array(l1-1,l2-1, 1) + (wcl(l3+1)*thrcof0(i)**2d0)
            end do
        end do
    end do

end subroutine


subroutine calc_cov_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                   & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                   & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                   & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                   & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                   & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                   & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                   & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                   & cov_array)


    implicit none
    real(8), intent(in)    :: TaTcTbTd(:), TaTdTbTc(:), PaPcPbPd(:), PaPdPbPc(:)
    real(8), intent(in)    :: TaTcPbPd(:), TaPdPbTc(:), PaPcTbTd(:), PaTdTbPc(:)
    real(8), intent(in)    :: TaPcTbPd(:), TaPdTbPc(:), TaTcTbPd(:), TaPdTbTc(:)
    real(8), intent(in)    :: TaPcTbTd(:), TaTdTbPc(:), TaPcPbTd(:), TaTdPbPc(:)
    real(8), intent(in)    :: TaPcPbPd(:), TaPdPbPc(:), PaPcTbPd(:), PaPdTbPc(:)
    real(8), intent(in)    :: TaTcPbTd(:), TaTdPbTc(:), PaTcTbTd(:), PaTdTbTc(:)
    real(8), intent(in)    :: PaTcPbTd(:), PaTdPbTc(:), PaTcTbPd(:), PaPdTbTc(:)
    real(8), intent(in)    :: PaTcPbPd(:), PaPdPbTc(:), PaPcPbTd(:), PaTdPbPc(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2)
    real(8) :: thrcof0(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,info,l1f,thrcof0,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i = l3-lmin+1
                cov_array(l1-1, l2-1, 1) = cov_array(l1-1, l2-1, 1) + (TaTcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 2) = cov_array(l1-1, l2-1, 2) + (TaTdTbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 3) = cov_array(l1-1, l2-1, 3) + (PaPcPbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 4) = cov_array(l1-1, l2-1, 4) + (PaPdPbPc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 5) = cov_array(l1-1, l2-1, 5) + (TaTcPbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 6) = cov_array(l1-1, l2-1, 6) + (TaPdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 7) = cov_array(l1-1, l2-1, 7) + (PaPcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 8) = cov_array(l1-1, l2-1, 8) + (PaTdTbPc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 9) = cov_array(l1-1, l2-1, 9) + (TaPcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 10) = cov_array(l1-1, l2-1, 10) + (TaPdTbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 11) = cov_array(l1-1, l2-1, 11) + (TaTcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 12) = cov_array(l1-1, l2-1, 12) + (TaPdTbTc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 13) = cov_array(l1-1, l2-1, 13) + (TaPcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 14) = cov_array(l1-1, l2-1, 14) + (TaTdTbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 15) = cov_array(l1-1, l2-1, 15) + (TaPcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 16) = cov_array(l1-1, l2-1, 16) + (TaTdPbPc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 17) = cov_array(l1-1, l2-1, 17) + (TaPcPbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 18) = cov_array(l1-1, l2-1, 18) + (TaPdPbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 19) = cov_array(l1-1, l2-1, 19) + (PaPcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 20) = cov_array(l1-1, l2-1, 20) + (PaPdTbPc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 21) = cov_array(l1-1, l2-1, 21) + (TaTcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 22) = cov_array(l1-1, l2-1, 22) + (TaTdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 23) = cov_array(l1-1, l2-1, 23) + (PaTcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 24) = cov_array(l1-1, l2-1, 24) + (PaTdTbTc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 25) = cov_array(l1-1, l2-1, 25) + (PaTcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 26) = cov_array(l1-1, l2-1, 26) + (PaTdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 27) = cov_array(l1-1, l2-1, 27) + (PaTcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 28) = cov_array(l1-1, l2-1, 28) + (PaPdTbTc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 29) = cov_array(l1-1, l2-1, 29) + (PaTcPbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 30) = cov_array(l1-1, l2-1, 30) + (PaPdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 31) = cov_array(l1-1, l2-1, 31) + (PaPcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 32) = cov_array(l1-1, l2-1, 32) + (PaTdPbPc(l3+1)*thrcof0(i)**2d0)

            end do
        end do
    end do

end subroutine


subroutine calc_cov_spin0and2_simple_planck(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                          & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                          & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                          & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                          & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                          & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                          & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                          & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                          & cov_array)


    implicit none
    real(8), intent(in)    :: TaTcTbTd(:), TaTdTbTc(:), PaPcPbPd(:), PaPdPbPc(:)
    real(8), intent(in)    :: TaTcPbPd(:), TaPdPbTc(:), PaPcTbTd(:), PaTdTbPc(:)
    real(8), intent(in)    :: TaPcTbPd(:), TaPdTbPc(:), TaTcTbPd(:), TaPdTbTc(:)
    real(8), intent(in)    :: TaPcTbTd(:), TaTdTbPc(:), TaPcPbTd(:), TaTdPbPc(:)
    real(8), intent(in)    :: TaPcPbPd(:), TaPdPbPc(:), PaPcTbPd(:), PaPdTbPc(:)
    real(8), intent(in)    :: TaTcPbTd(:), TaTdPbTc(:), PaTcTbTd(:), PaTdTbTc(:)
    real(8), intent(in)    :: PaTcPbTd(:), PaTdPbTc(:), PaTcTbPd(:), PaPdTbTc(:)
    real(8), intent(in)    :: PaTcPbPd(:), PaPdPbTc(:), PaPcPbTd(:), PaTdPbPc(:)
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, l3, info, nlmax, lmin, lmax, i
    real(8) :: l1f(2)
    real(8) :: thrcof0(2*size(cov_array,1)),thrcof1(2*size(cov_array,1))
    nlmax = size(cov_array,1)-1
    !$omp parallel do private(l3,l2,l1,info,l1f,thrcof0,thrcof1,lmin,lmax,i) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = l1, nlmax
            call drc3jj(dble(l1),dble(l2),0d0,0d0,l1f(1),l1f(2),thrcof0, size(thrcof0),info)
            call drc3jj(dble(l1),dble(l2),-2d0,2d0,l1f(1),l1f(2),thrcof1, size(thrcof1),info)

            lmin=INT(l1f(1))
            lmax=MIN(nlmax+1,INT(l1f(2)))
            do l3=lmin,lmax
                i = l3-lmin+1
                cov_array(l1-1, l2-1, 1) = cov_array(l1-1, l2-1, 1) + (TaTcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 2) = cov_array(l1-1, l2-1, 2) + (TaTdTbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 3) = cov_array(l1-1, l2-1, 3) + (PaPcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 4) = cov_array(l1-1, l2-1, 4) + (PaPdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)

                cov_array(l1-1, l2-1, 5) = cov_array(l1-1, l2-1, 5) + (TaTcPbPd(l3+1)*thrcof0(i)*thrcof1(i))
                cov_array(l1-1, l2-1, 6) = cov_array(l1-1, l2-1, 6) + (TaPdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 7) = cov_array(l1-1, l2-1, 7) + (PaPcTbTd(l3+1)*thrcof0(i)*thrcof1(i))
                cov_array(l1-1, l2-1, 8) = cov_array(l1-1, l2-1, 8) + (PaTdTbPc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 9) = cov_array(l1-1, l2-1, 9) + (TaPcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 10) = cov_array(l1-1, l2-1, 10) + (TaPdTbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 11) = cov_array(l1-1, l2-1, 11) + (TaTcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 12) = cov_array(l1-1, l2-1, 12) + (TaPdTbTc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 13) = cov_array(l1-1, l2-1, 13) + (TaPcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 14) = cov_array(l1-1, l2-1, 14) + (TaTdTbPc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 15) = cov_array(l1-1, l2-1, 15) + (TaPcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 16) = cov_array(l1-1, l2-1, 16) + (TaTdPbPc(l3+1)*thrcof0(i)*thrcof1(i))

                cov_array(l1-1, l2-1, 17) = cov_array(l1-1, l2-1, 17) + (TaPcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 18) = cov_array(l1-1, l2-1, 18) + (TaPdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 19) = cov_array(l1-1, l2-1, 19) + (PaPcTbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 20) = cov_array(l1-1, l2-1, 20) + (PaPdTbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)

                cov_array(l1-1, l2-1, 21) = cov_array(l1-1, l2-1, 21) + (TaTcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 22) = cov_array(l1-1, l2-1, 22) + (TaTdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 23) = cov_array(l1-1, l2-1, 23) + (PaTcTbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 24) = cov_array(l1-1, l2-1, 24) + (PaTdTbTc(l3+1)*thrcof0(i)**2d0)

                cov_array(l1-1, l2-1, 25) = cov_array(l1-1, l2-1, 25) + (PaTcPbTd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 26) = cov_array(l1-1, l2-1, 26) + (PaTdPbTc(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 27) = cov_array(l1-1, l2-1, 27) + (PaTcTbPd(l3+1)*thrcof0(i)**2d0)
                cov_array(l1-1, l2-1, 28) = cov_array(l1-1, l2-1, 28) + (PaPdTbTc(l3+1)*thrcof0(i)*thrcof1(i))

                cov_array(l1-1, l2-1, 29) = cov_array(l1-1, l2-1, 29) + (PaTcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 30) = cov_array(l1-1, l2-1, 30) + (PaPdPbTc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 31) = cov_array(l1-1, l2-1, 31) + (PaPcPbTd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
                cov_array(l1-1, l2-1, 32) = cov_array(l1-1, l2-1, 32) + (PaTdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)

            end do
        end do
    end do

end subroutine






