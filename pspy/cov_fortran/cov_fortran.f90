module cov_compute

contains

subroutine calc_cov_elem_spin0(ac_bd, ad_bc, l1, l2, nlmax, elems)
    implicit none
    integer, intent(in)   :: l1, l2, nlmax
    real(8), intent(in)   ::  ac_bd(:),ad_bc(:)
    real(8), intent(inout):: elems(2)
    real(8) :: thrcof0(2*(nlmax+1)), l1f, l2f
    integer :: info, l3, wlmin, wlmax, i
    call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)
    wlmin = int(l1f)
    wlmax = min(nlmax+1, int(l2f))
    elems = 0
    do l3 = wlmin, wlmax
        i = l3 - wlmin + 1
        elems(1) = elems(1) + ac_bd(l3+1)*thrcof0(i)**2
        elems(2) = elems(2) + ad_bc(l3+1)*thrcof0(i)**2
    end do
end subroutine

subroutine calc_cov_elem_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                        & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                        & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                        & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                        & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                        & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                        & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                        & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                        doPlanck, l1, l2, nlmax, elems)
    implicit none
    integer, intent(in)    :: doPlanck, l1, l2, nlmax
    real(8), intent(in)    :: TaTcTbTd(:), TaTdTbTc(:), PaPcPbPd(:), PaPdPbPc(:)
    real(8), intent(in)    :: TaTcPbPd(:), TaPdPbTc(:), PaPcTbTd(:), PaTdTbPc(:)
    real(8), intent(in)    :: TaPcTbPd(:), TaPdTbPc(:), TaTcTbPd(:), TaPdTbTc(:)
    real(8), intent(in)    :: TaPcTbTd(:), TaTdTbPc(:), TaPcPbTd(:), TaTdPbPc(:)
    real(8), intent(in)    :: TaPcPbPd(:), TaPdPbPc(:), PaPcTbPd(:), PaPdTbPc(:)
    real(8), intent(in)    :: TaTcPbTd(:), TaTdPbTc(:), PaTcTbTd(:), PaTdTbTc(:)
    real(8), intent(in)    :: PaTcPbTd(:), PaTdPbTc(:), PaTcTbPd(:), PaPdTbTc(:)
    real(8), intent(in)    :: PaTcPbPd(:), PaPdPbTc(:), PaPcPbTd(:), PaTdPbPc(:)
    real(8), intent(inout):: elems(32)
    real(8) :: thrcof0(2*(nlmax+1)),thrcof1(2*(nlmax+1)), l1f, l2f
    integer :: info, l3, wlmin, wlmax, i

    if (doPlanck .eq. 0) then
        call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)
        wlmin = int(l1f)
        wlmax = min(nlmax+1, int(l2f))
        elems = 0
        do l3 = wlmin, wlmax
            i = l3 - wlmin + 1
            elems(1) = elems(1) + (TaTcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(2) = elems(2) + (TaTdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(3) = elems(3) + (PaPcPbPd(l3+1)*thrcof0(i)**2d0)
            elems(4) = elems(4) + (PaPdPbPc(l3+1)*thrcof0(i)**2d0)
            elems(5) = elems(5) + (TaTcPbPd(l3+1)*thrcof0(i)**2d0)
            elems(6) = elems(6) + (TaPdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(7) = elems(7) + (PaPcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(8) = elems(8) + (PaTdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(9) = elems(9) + (TaPcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(10) = elems(10) + (TaPdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(11) = elems(11) + (TaTcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(12) = elems(12) + (TaPdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(13) = elems(13) + (TaPcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(14) = elems(14) + (TaTdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(15) = elems(15) + (TaPcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(16) = elems(16) + (TaTdPbPc(l3+1)*thrcof0(i)**2d0)
            elems(17) = elems(17) + (TaPcPbPd(l3+1)*thrcof0(i)**2d0)
            elems(18) = elems(18) + (TaPdPbPc(l3+1)*thrcof0(i)**2d0)
            elems(19) = elems(19) + (PaPcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(20) = elems(20) + (PaPdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(21) = elems(21) + (TaTcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(22) = elems(22) + (TaTdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(23) = elems(23) + (PaTcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(24) = elems(24) + (PaTdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(25) = elems(25) + (PaTcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(26) = elems(26) + (PaTdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(27) = elems(27) + (PaTcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(28) = elems(28) + (PaPdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(29) = elems(29) + (PaTcPbPd(l3+1)*thrcof0(i)**2d0)
            elems(30) = elems(30) + (PaPdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(31) = elems(31) + (PaPcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(32) = elems(32) + (PaTdPbPc(l3+1)*thrcof0(i)**2d0)
        end do
    else
        call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)
        call drc3jj(dble(l1), dble(l2), -2d0, 2d0, l1f, l2f, thrcof1, size(thrcof1), info)

        wlmin = int(l1f)
        wlmax = min(nlmax+1, int(l2f))
        elems = 0
        do l3 = wlmin, wlmax
            i = l3 - wlmin + 1
            elems(1) = elems(1) + (TaTcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(2) = elems(2) + (TaTdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(3) = elems(3) + (PaPcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(4) = elems(4) + (PaPdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(5) = elems(5) + (TaTcPbPd(l3+1)*thrcof0(i)*thrcof1(i))
            elems(6) = elems(6) + (TaPdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(7) = elems(7) + (PaPcTbTd(l3+1)*thrcof0(i)*thrcof1(i))
            elems(8) = elems(8) + (PaTdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(9) = elems(9) + (TaPcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(10) = elems(10) + (TaPdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(11) = elems(11) + (TaTcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(12) = elems(12) + (TaPdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(13) = elems(13) + (TaPcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(14) = elems(14) + (TaTdTbPc(l3+1)*thrcof0(i)**2d0)
            elems(15) = elems(15) + (TaPcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(16) = elems(16) + (TaTdPbPc(l3+1)*thrcof0(i)*thrcof1(i))
            elems(17) = elems(17) + (TaPcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(18) = elems(18) + (TaPdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(19) = elems(19) + (PaPcTbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(20) = elems(20) + (PaPdTbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(21) = elems(21) + (TaTcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(22) = elems(22) + (TaTdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(23) = elems(23) + (PaTcTbTd(l3+1)*thrcof0(i)**2d0)
            elems(24) = elems(24) + (PaTdTbTc(l3+1)*thrcof0(i)**2d0)
            elems(25) = elems(25) + (PaTcPbTd(l3+1)*thrcof0(i)**2d0)
            elems(26) = elems(26) + (PaTdPbTc(l3+1)*thrcof0(i)**2d0)
            elems(27) = elems(27) + (PaTcTbPd(l3+1)*thrcof0(i)**2d0)
            elems(28) = elems(28) + (PaPdTbTc(l3+1)*thrcof0(i)*thrcof1(i))
            elems(29) = elems(29) + (PaTcPbPd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(30) = elems(30) + (PaPdPbTc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(31) = elems(31) + (PaPcPbTd(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)
            elems(32) = elems(32) + (PaTdPbPc(l3+1)*thrcof1(i)**2*(1+(-1)**(l1+l2+l3))/2)

        end do
    end if

end subroutine





subroutine calc_cov_spin0(ac_bd, ad_bc, l_exact, l_band, l_toeplitz, cov_array)
    implicit none
    real(8), intent(in)    :: ac_bd(:),ad_bc(:)
    integer, intent(in)   :: l_exact, l_band, l_toeplitz
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, nlmax, lmax_band
    nlmax = size(cov_array,1)-1

    !$omp parallel do private(l2, l1) schedule(dynamic)
    do l1 = 2, min(nlmax, l_exact)
        do l2 = l1, nlmax
            call calc_cov_elem_spin0(ac_bd, ad_bc, l1, l2, nlmax, cov_array(l1-1, l2-1,:))
        end do
    end do

    if (l_exact .lt. nlmax) then

        !$omp parallel do private(l2, l1) schedule(dynamic)
        do l1 = l_exact+1, l_toeplitz

            if (l1 .lt. l_toeplitz) then
                lmax_band = min(l1 + l_band, nlmax)
            else
                lmax_band = nlmax
            end if

            do l2 = l1, lmax_band
                call calc_cov_elem_spin0(ac_bd, ad_bc, l1, l2, nlmax, cov_array(l1-1, l2-1,:))
            end do
        end do

    	if (l_toeplitz .lt. nlmax) then
            !$omp parallel do private(l1) schedule(dynamic)
            do l1 = l_toeplitz+1, nlmax
                call calc_cov_elem_spin0(ac_bd, ad_bc, l1, l1, nlmax, cov_array(l1-1, l1-1,:))
            end do
        end if
    end if

end subroutine


subroutine calc_cov_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                   & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                   & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                   & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                   & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                   & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                   & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                   & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                   & doPlanck, l_exact, l_band, l_toeplitz, cov_array)


    implicit none
    real(8), intent(in)    :: TaTcTbTd(:), TaTdTbTc(:), PaPcPbPd(:), PaPdPbPc(:)
    real(8), intent(in)    :: TaTcPbPd(:), TaPdPbTc(:), PaPcTbTd(:), PaTdTbPc(:)
    real(8), intent(in)    :: TaPcTbPd(:), TaPdTbPc(:), TaTcTbPd(:), TaPdTbTc(:)
    real(8), intent(in)    :: TaPcTbTd(:), TaTdTbPc(:), TaPcPbTd(:), TaTdPbPc(:)
    real(8), intent(in)    :: TaPcPbPd(:), TaPdPbPc(:), PaPcTbPd(:), PaPdTbPc(:)
    real(8), intent(in)    :: TaTcPbTd(:), TaTdPbTc(:), PaTcTbTd(:), PaTdTbTc(:)
    real(8), intent(in)    :: PaTcPbTd(:), PaTdPbTc(:), PaTcTbPd(:), PaPdTbTc(:)
    real(8), intent(in)    :: PaTcPbPd(:), PaPdPbTc(:), PaPcPbTd(:), PaTdPbPc(:)
    integer, intent(in)    :: l_exact, l_band, l_toeplitz, doPlanck
    real(8), intent(inout) :: cov_array(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, nlmax, lmax_band
    nlmax = size(cov_array,1)-1

    !$omp parallel do private(l2, l1) schedule(dynamic)
    do l1 = 2, min(nlmax, l_exact)
        do l2 = l1, nlmax
            call calc_cov_elem_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                              & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                              & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                              & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                              & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                              & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                              & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                              & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                              doPlanck, l1, l2, nlmax, cov_array(l1-1, l2-1, :))
        end do
    end do

    if (l_exact .lt. nlmax) then
        !$omp parallel do private(l2, l1, lmax_band) schedule(dynamic)
        do l1 = l_exact + 1, l_toeplitz
            if (l1 .lt. l_toeplitz) then
                lmax_band = min(l1 + l_band, nlmax)
            else
                lmax_band = nlmax
            end if
            do l2 = l1, lmax_band
                call calc_cov_elem_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                                  & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                                  & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                                  & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                                  & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                                  & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                                  & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                                  & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                                  doPlanck, l1, l2, nlmax, cov_array(l1-1, l2-1, :))

            end do
        end do

        if (l_toeplitz .lt. nlmax) then
            !$omp parallel do private(l1) schedule(dynamic)
            do l1 = l_toeplitz + 1, nlmax
                call calc_cov_elem_spin0and2_simple(TaTcTbTd, TaTdTbTc, PaPcPbPd, PaPdPbPc, &
                                                  & TaTcPbPd, TaPdPbTc, PaPcTbTd, PaTdTbPc, &
                                                  & TaPcTbPd, TaPdTbPc, TaTcTbPd, TaPdTbTc, &
                                                  & TaPcTbTd, TaTdTbPc, TaPcPbTd, TaTdPbPc, &
                                                  & TaPcPbPd, TaPdPbPc, PaPcTbPd, PaPdTbPc, &
                                                  & TaTcPbTd, TaTdPbTc, PaTcTbTd, PaTdTbTc, &
                                                  & PaTcPbTd, PaTdPbTc, PaTcTbPd, PaPdTbTc, &
                                                  & PaTcPbPd, PaPdPbTc, PaPcPbTd, PaTdPbPc, &
                                                  doPlanck, l1, l1, nlmax, cov_array(l1-1, l1-1, :))

            end do
        end if
    end if

end subroutine

end module
