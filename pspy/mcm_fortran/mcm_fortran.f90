module mcm_compute

contains

subroutine calc_coupling_elem_spin0(wcl, l1, l2, elem)
    implicit none
    integer, intent(in)   :: l1, l2
    real(8), intent(in)   :: wcl(:)
    real(8), intent(inout):: elem
    real(8) :: thrcof0(2*size(wcl)), l1f, l2f
    integer :: info, l3, wlmin, wlmax, i, lmax
    lmax = size(wcl)-1 ! wcl starts at 0

    call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)

    wlmin = int(l1f)
    wlmax = min(lmax, int(l2f))
    elem = 0
    do l3 = wlmin, wlmax
        i = l3 - wlmin + 1
        elem = elem + wcl(l3 + 1) * thrcof0(i)**2
    end do
end subroutine

subroutine calc_coupling_elem_spin0and2(wcl_00, wcl_02, wcl_20, wcl_22, l1, l2, elems)
    implicit none
    integer, intent(in)   :: l1, l2
    real(8), intent(in)   :: wcl_00(:), wcl_02(:), wcl_20(:), wcl_22(:)
    real(8), intent(inout):: elems(5)
    real(8) :: thrcof0(2*size(wcl_00)), thrcof1(2*size(wcl_00)), l1f, l2f
    integer :: info, l3, wlmin, wlmax, i, lmax

    lmax = size(wcl_00)-1 ! wcl starts at 0

    call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)
    call drc3jj(dble(l1), dble(l2), -2d0, 2d0, l1f, l2f, thrcof1, size(thrcof1), info)

    wlmin = int(l1f)
    wlmax = min(lmax, int(l2f))
    elems = 0
    do l3 = wlmin, wlmax
        i = l3 - wlmin + 1
        elems(1) = elems(1) + wcl_00(l3 + 1) * thrcof0(i)**2
        elems(2) = elems(2) + wcl_02(l3 + 1) * thrcof0(i) * thrcof1(i)
        elems(3) = elems(3) + wcl_20(l3 + 1) * thrcof0(i) * thrcof1(i)
        elems(4) = elems(4) + wcl_22(l3 + 1) * thrcof1(i)**2 * (1 + (-1)**(l1 + l2 + l3)) / 2
        elems(5) = elems(5) + wcl_22(l3 + 1) * thrcof1(i)**2 * (1 - (-1)**(l1 + l2 + l3)) / 2
    end do
end subroutine

subroutine calc_coupling_elem_spin0and2_pure(wcl_00, wcl_02, wcl_20, wcl_22, l1, l2, elems)
    implicit none
    integer, intent(in)   :: l1, l2
    real(8), intent(in)   :: wcl_00(:), wcl_02(:), wcl_20(:), wcl_22(:)
    real(8), intent(inout):: elems(5)
    real(8) :: thrcof0(2*size(wcl_00)), thrcofa(2*size(wcl_00)), thrcofb(2*size(wcl_00)), thrcofc(2*size(wcl_00)), l1f, l2f, fac_b, fac_c, combin
    integer :: info, l3, wlmin1, wlmax1, i1, wlmin2, i2, wlmin3, i3, lmax

    lmax = size(wcl_00)-1 ! wcl starts at 0

    call drc3jj(dble(l1), dble(l2), 0d0, 0d0, l1f, l2f, thrcof0, size(thrcof0), info)
    call drc3jj(dble(l1), dble(l2), -2d0, 2d0, l1f, l2f, thrcofa, size(thrcofa), info)
    wlmin1 = int(l1f)
    wlmax1 = min(lmax, int(l2f))
    call drc3jj(dble(l1), dble(l2), -2d0, 1d0, l1f, l2f, thrcofb, size(thrcofb), info)
    wlmin2 = int(l1f)
    call drc3jj(dble(l1), dble(l2), -2d0, 0d0, l1f, l2f, thrcofc, size(thrcofc), info)
    wlmin3 = int(l1f)
    elems = 0
    do l3 = wlmin1, wlmax1
        i1   = l3 - wlmin1 + 1
        i2   = l3 - wlmin2 + 1
        i3   = l3 - wlmin3 + 1
        fac_b = 2 * dsqrt((l3 + 1d0) * l3 / ((l2 - 1d0) * (l2 + 2d0)))
        fac_c = dsqrt((l3 + 2d0) * (l3 + 1d0) * l3 * (l3 - 1d0) / ((l2 + 2d0) * (l2 + 1d0) * l2 * (l2 - 1d0)))

        if (i2 < 0) then
            fac_b = 0d0
        end if

        if (i3 < 0) then
            fac_c = 0d0
        end if

        combin = thrcofa(i1) + fac_b * thrcofb(i2) + fac_c * thrcofc(i3)
        elems(1) = elems(1) + (wcl_00(l3 + 1) * thrcof0(i1)**2d0)
        elems(2) = elems(2) + (wcl_02(l3 + 1) * thrcof0(i1) * combin)
        elems(3) = elems(3) + (wcl_20(l3 + 1) * thrcof0(i1) * combin)
        elems(4) = elems(4) + (wcl_22(l3 + 1) * combin**2 * (1 + (-1)**(l1 + l2 + l3)) / 2)
        elems(5) = elems(5) + (wcl_22(l3 + 1) * combin**2 * (1 - (-1)**(l1 + l2 + l3)) / 2)
    
    end do
end subroutine

 

subroutine calc_coupling_spin0(wcl, l_exact, l_band, l_toeplitz, coupling)
    implicit none
    real(8), intent(in)    :: wcl(:)
    integer, intent(in)    :: l_exact , l_band, l_toeplitz
    real(8), intent(inout) :: coupling(:,:)
    integer :: l1, l2, nlmax, lmax_band

    nlmax = size(coupling,1) - 1

    !$omp parallel do private(l2, l1) schedule(dynamic)
    do l1 = 2, min(nlmax, l_exact)
        do l2 = l1, nlmax
            call calc_coupling_elem_spin0(wcl, l1, l2, coupling(l1-1, l2-1))
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
                call calc_coupling_elem_spin0(wcl, l1, l2, coupling(l1-1, l2-1))
            end do
        end do

        if (l_toeplitz .lt. nlmax) then
            !$omp parallel do
            do l1 = l_toeplitz + 1, nlmax
                call calc_coupling_elem_spin0(wcl, l1, l1, coupling(l1-1, l1-1))
            end do
        end if
    end if

end subroutine

subroutine calc_coupling_spin0and2(wcl_00, wcl_02, wcl_20, wcl_22, l_exact, l_band, l_toeplitz, coupling)
    implicit none
    real(8), intent(in)    :: wcl_00(:), wcl_02(:), wcl_20(:), wcl_22(:)
    integer, intent(in)   :: l_exact, l_band, l_toeplitz
    real(8), intent(inout) :: coupling(:,:,:)
    real(8), parameter     :: pi = 3.14159265358979323846264d0
    integer :: l1, l2, nlmax, lmax_band
    nlmax = size(coupling, 1) - 1

    !$omp parallel do private(l2, l1) schedule(dynamic)
    do l1 = 2, min(nlmax, l_exact)
        do l2 = l1, nlmax
            call calc_coupling_elem_spin0and2(wcl_00, wcl_02, wcl_20, wcl_22, l1, l2, coupling(l1-1, l2-1, :))
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
                call calc_coupling_elem_spin0and2(wcl_00, wcl_02, wcl_20, wcl_22, l1, l2, coupling(l1-1, l2-1, :))
            end do
        end do

        if (l_toeplitz .lt. nlmax) then
            !$omp parallel do 
            do l1 = l_toeplitz + 1, nlmax
                call calc_coupling_elem_spin0and2(wcl_00, wcl_02, wcl_20, wcl_22, l1, l1, coupling(l1-1, l1-1, :))
            end do
        end if
    end if

end subroutine


subroutine calc_mcm_spin0and2_pure(wcl_00, wcl_02, wcl_20, wcl_22, coupling_array)
    implicit none
    real(8), intent(in)    :: wcl_00(:), wcl_02(:), wcl_20(:), wcl_22(:)
    real(8), intent(inout) :: coupling_array(:,:,:)
    integer :: l1, l2, nlmax
    nlmax = size(coupling_array,1)-1
    !$omp parallel do private(l2,l1) schedule(dynamic)
    do l1 = 2, nlmax
        do l2 = 2, nlmax
            call calc_coupling_elem_spin0and2_pure(wcl_00, wcl_02, wcl_20, wcl_22, l1, l2, coupling_array(l1-1,l2-1,:))
        end do
    end do

end subroutine

subroutine toepliz_array_fortran(toepliz_array, coupling_array, l_toep)
    implicit none
    integer, intent(in)    :: l_toep
    real(8), intent(inout) :: toepliz_array(:,:)
    real(8), intent(in) :: coupling_array(:,:)
    real(8) :: diag(size(coupling_array,1)), row(size(coupling_array,1))
    integer :: l1, l2, nlmax

    nlmax = size(coupling_array, 1) - 1

    !$omp parallel do
    do l1 = 2, nlmax
        diag(l1-1) = coupling_array(l1-1, l1-1)**0.5d0
    end do

    !$omp parallel do
    do l2 = l_toep, nlmax
        row(l2 - l_toep + 1)= coupling_array(l_toep-1,l2-1)/(diag(l_toep-1)*diag(l2-1))
    end do

    !$omp parallel do private(l2,l1) schedule(dynamic)
    do l1 = 2, l_toep
        do l2 = l1,  l1 + nlmax - l_toep
            toepliz_array(l1-1, l2-1) = row(l2-l1+1) * (diag(l1-1)*diag(l2-1))
        end do
    end do

  !$omp parallel do private(l2,l1) schedule(dynamic)
   do l1 = l_toep + 1, nlmax
        do l2 = l1, nlmax
            toepliz_array(l1-1, l2-1) = row(l2-l1+1) * (diag(l1-1)*diag(l2-1))
        end do
  end do

end subroutine


subroutine toepliz_array_fortran2(toepliz_array, coupling_array, l_toep, l_exact)
    implicit none
    integer, intent(in)    :: l_toep, l_exact
    real(8), intent(inout) :: toepliz_array(:,:)
    real(8), intent(in) :: coupling_array(:,:)
    real(8) :: diag(size(coupling_array,1)), row_toep(size(coupling_array,1)), row_exact(size(coupling_array,1))
    integer :: l1, l2, nlmax

    nlmax = size(coupling_array, 1) - 1

    !$omp parallel do
    do l1 = 2, nlmax
        diag(l1-1) = coupling_array(l1-1, l1-1)**0.5d0
    end do

    !$omp parallel do
    do l2 = l_toep, nlmax
        row_toep(l2 - l_toep + 1)= coupling_array(l_toep-1,l2-1)/(diag(l_toep-1)*diag(l2-1))
    end do

    !$omp parallel do
    do l2 = l_exact, nlmax
        row_exact(l2 - l_exact + 1) = coupling_array(l_exact-1,l2-1)/(diag(l_exact-1)*diag(l2-1))
    end do

    !$omp parallel do private(l2,l1) schedule(dynamic)
    do l1 = 2, l_toep
        do l2 = l1,  l1 + nlmax - l_toep
            toepliz_array(l1-1, l2-1) = row_toep(l2-l1+1) * (diag(l1-1)*diag(l2-1))
        end do
        do l2 = l1 + nlmax - l_toep + 1,   nlmax
            toepliz_array(l1-1, l2-1) = row_exact(l2-l1+1) * (diag(l1-1)*diag(l2-1))
        end do
    end do

  !$omp parallel do private(l2,l1) schedule(dynamic)
   do l1 = l_toep + 1, nlmax
        do l2 = l1, nlmax
            toepliz_array(l1-1, l2-1) = row_toep(l2-l1+1) * (diag(l1-1)*diag(l2-1))
        end do
  end do

end subroutine


subroutine fill_upper(mat)
  implicit none
  real(8), intent(inout) :: mat(:,:)
  integer ::  j, k
  !$omp parallel do private(j,k) schedule(dynamic)
    do j = 1, size(mat,2)
      do k = 1, j-1
        mat(j,k) = mat(k,j)
      end do
    end do
end subroutine



subroutine bin_mcm(mcm, binLo, binHi, binsize, mbb, doDl)
    ! Bin the given mode coupling matrix mcm(0:lmax,0:lmax) into
    ! mbb(nbin,nbin) using bins of the given binsize
    implicit none
    real(8), intent(in)    :: mcm(:,:)
    integer, intent(in)    :: binLo(:),binHi(:),binsize(:),doDl
    real(8), intent(inout) :: mbb(:,:)
    integer :: b1, b2, l1, l2, lmax
    lmax = size(mcm,1)-1
    mbb  = 0
    do b2=1,size(mbb,1)
        do b1=1,size(mbb,1)
            do l2=binLo(b2),binHi(b2)
                do l1=binLo(b1),binHi(b1)
                    if (doDl .eq. 1) then
                        mbb(b1,b2)=mbb(b1,b2) + mcm(l1-1,l2-1)*l2*(l2+1d0)/(l1*(l1+1d0)) !*mcm(l2-1,l3-1)
                    else
                        mbb(b1,b2)=mbb(b1,b2) + mcm(l1-1,l2-1)
                    end if
                end do
            end do
            mbb(b1,b2) = mbb(b1,b2) / binsize(b2)

        end do
    end do
end subroutine

subroutine binning_matrix(mcm, binLo, binHi, binsize, bbl, doDl)
    implicit none
    real(8), intent(in)    :: mcm(:,:)
    integer(8), intent(in)    :: binLo(:),binHi(:),binsize(:),doDl
    real(8), intent(inout) :: bbl(:,:)
    integer(8) :: b2, l1, l2,lmax

    lmax = size(mcm,1)-1
    ! mcm is transposed
    ! compute \sum_{l'} M_l'l
    do l1=2,lmax
        do b2=1,size(binLo)
            do l2=binLo(b2),binHi(b2)
                if (doDl .eq. 1) then
                    bbl(l1-1,b2)=bbl(l1-1,b2)+mcm(l1-1,l2-1)*l2*(l2+1d0)/(l1*(l1+1d0))
                else
                     bbl(l1-1,b2)=bbl(l1-1,b2)+mcm(l1-1,l2-1)
                end if
            end do
            bbl(l1-1,b2)=bbl(l1-1,b2)/(binsize(b2)*1d0)
        end do
    end do
end subroutine


end module

