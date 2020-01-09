# pure BB spatial window follwoing Grain+ 2009
# will check a couple things and add comments soon -Steve 1/24/19
#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pixell import sharp,curvedsky,enmap
import copy
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils
import os
from astropy.io import fits
import pymaster as nmt
from copy import deepcopy



#We start by specifying the CAR survey parameters, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
# It will have a resolution of 1 arcminute
ra0,ra1,dec0,dec1=-25,25,-25,25
res=3
#We then specify the HEALPIX survey parameter, it will be a disk of radius 25 degree centered on longitude 30 degree and latitude 50 degree
# It will have a resolution nside=1024
lon,lat=30,50
radius=25
nside=512
# ncomp=3 mean that we are going to use spin0 and 2 field
ncomp=3
# specify the order of the spectra, this will be the order used in pspy
# note that if you are doing cross correlation between galaxy and kappa for example, you should follow a similar structure
spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
# clfile are the camb lensed power spectra
clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat'
# nSplits stand for the number of splits we want to simulate
nSplits=1
# a binningfile with format, lmin,lmax,lmean
binning_file='../data/binningFile_100_50.dat'
# the maximum multipole to consider
lmax=3*nside-1
# the number of iteration in map2alm
niter=0
# the noise on the spin0 component, if not specified, the noise in polarisation wil be sqrt(2)x that
rms_uKarcmin_T=1
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey=5

type='Cl'

# the templates for the CMB splits
template_car= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
template_healpix= so_map.healpix_template(ncomp,nside=nside)

# the binary template for the window functionpixels
# for CAR we set pixels inside the survey at 1 and  at the border to be zero
binary_car=so_map.car_template(1,ra0,ra1,dec0,dec1,res)
binary_car.data[:]=0
binary_car.data[1:-1,1:-1]=1

# for Healpix we set pixel inside the disk at 1 and pixel outside at zero
binary_healpix=so_map.healpix_template(ncomp=1,nside=nside)
vec=hp.pixelfunc.ang2vec(lon,lat, lonlat=True)
disc=hp.query_disc(nside, vec, radius=radius*np.pi/180)
binary_healpix.data[disc]=1

# for the CAR survey we will use an apodisation type designed for rectangle maps, and a C1 apodisation for the healpix maps
apo_type_list=['C1']
binary_list=[binary_healpix]
template_list=[template_healpix]
run_name=['run_HEALPIX']

# Ok let's loop on the CAR and HEALPIX case, starting with CAR


test_dir='result_spectra_spin0and2_pure_namaster'
try:
    os.makedirs(test_dir)
except:
    pass


Namaster=True
iStart=0
iStop=100

for run,apo_type,binary,template in zip(run_name,apo_type_list,binary_list,template_list):
    print ( 'start %s'%run)
    
    window=so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree_survey)
    
    mcm,mbb_inv_pure,Bbl_pure=so_mcm.mcm_and_bbl_spin0and2((window,window), binning_file, lmax=lmax,niter=niter, type=type,pure=True,return_mcm=True)
    mcm,mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0and2((window,window), binning_file, lmax=lmax,niter=niter, type=type,pure=False,return_mcm=True)

    cmb=template.synfast(clfile)
    if Namaster:
        fyp=nmt.NmtField(window.data,[cmb.data[1],cmb.data[2]],purify_e=True,purify_b=True,n_iter_mask_purify=0,n_iter=0)
        fynp=nmt.NmtField(window.data,[cmb.data[1],cmb.data[2]],n_iter_mask_purify=0,n_iter=0)
        b=nmt.NmtBin(nside,nlb=50)
        w_yp=nmt.NmtWorkspace(); w_yp.compute_coupling_matrix(fyp,fyp,b)
        w_ynp=nmt.NmtWorkspace(); w_ynp.compute_coupling_matrix(fynp,fynp,b)
        leff=b.get_effective_ells()

        def compute_master(f_a,f_b,wsp) :
            cl_coupled=nmt.compute_coupled_cell(f_a,f_b)
            cl_decoupled=wsp.decouple_cell(cl_coupled)
            return cl_decoupled

        list_EE_namaster_pure=[]
        list_BB_namaster_pure=[]
        list_EE_namaster=[]
        list_BB_namaster=[]

    list_EE=[]
    list_BB=[]
    list_EE_pure=[]
    list_BB_pure=[]
    for i in range(iStart,iStop):
        print(i)
        cmb=template.synfast(clfile)
        
        alm= sph_tools.get_alms(cmb,(window,window),niter,lmax)
        alm_pure = sph_tools.get_pure_alms(cmb,(window,window),niter,lmax)

        l,ps_pure= so_spectra.get_spectra(alm_pure,alm_pure,spectra=spectra)
        l,ps= so_spectra.get_spectra(alm,alm,spectra=spectra)

        lb,Db_dict_pure=so_spectra.bin_spectra(l,ps_pure,binning_file,lmax,type=type,mbb_inv=mbb_inv_pure,spectra=spectra)
        lb,Db_dict=so_spectra.bin_spectra(l,ps,binning_file,lmax,type=type,mbb_inv=mbb_inv,spectra=spectra)
        
        so_spectra.write_ps('%s/spectra_pure_%s_%03d.dat'%(test_dir,run,i),lb,Db_dict_pure,type=type,spectra=spectra)
        so_spectra.write_ps('%s/spectra_%s_%03d.dat'%(test_dir,run,i),lb,Db_dict,type=type,spectra=spectra)

        
        if Namaster:
            fyp=nmt.NmtField(window.data,[cmb.data[1],cmb.data[2]],purify_e=True,purify_b=True,n_iter_mask_purify=0,n_iter=0)
            fynp=nmt.NmtField(window.data,[cmb.data[1],cmb.data[2]],n_iter_mask_purify=0,n_iter=0)
            data=compute_master(fyp,fyp,w_yp)
            data_np=compute_master(fynp,fynp,w_ynp)
            
            Db_dict_pure_namaster=deepcopy(Db_dict_pure)
            Db_dict_namaster=deepcopy(Db_dict)
            Db_dict_pure_namaster['EE']=data[0]
            Db_dict_pure_namaster['BB']=data[3]
            Db_dict_namaster['EE']=data_np[0]
            Db_dict_namaster['BB']=data_np[3]
            so_spectra.write_ps('%s/spectra_pure_namaster_%s_%03d.dat'%(test_dir,run,i),lb,Db_dict_pure_namaster,type=type,spectra=spectra)
            so_spectra.write_ps('%s/spectra_namaster_%s_%03d.dat'%(test_dir,run,i),lb,Db_dict_namaster,type=type,spectra=spectra)


            list_EE_namaster_pure+=[data[0]]
            list_BB_namaster_pure+=[data[3]]
            list_EE_namaster+=[data_np[0]]
            list_BB_namaster+=[data_np[3]]


        list_EE_pure+=[Db_dict_pure['EE']]
        list_BB_pure+=[Db_dict_pure['BB']]
        list_EE+=[Db_dict['EE']]
        list_BB+=[Db_dict['BB']]


    std_EE=np.std(list_EE,axis=0)
    std_BB=np.std(list_BB,axis=0)
    meanEE=np.mean(list_EE,axis=0)
    meanBB=np.mean(list_BB,axis=0)

    std_EE_pure=np.std(list_EE_pure,axis=0)
    std_BB_pure=np.std(list_BB_pure,axis=0)
    meanEE_pure=np.mean(list_EE_pure,axis=0)
    meanBB_pure=np.mean(list_BB_pure,axis=0)

    if Namaster:
        std_EE_pure_namaster=np.std(list_EE_namaster_pure,axis=0)
        std_BB_pure_namaster=np.std(list_BB_namaster_pure,axis=0)
        meanEE_pure_namaster=np.mean(list_EE_namaster_pure,axis=0)
        meanBB_pure_namaster=np.mean(list_BB_namaster_pure,axis=0)

        std_EE_namaster=np.std(list_EE_namaster,axis=0)
        std_BB_namaster=np.std(list_BB_namaster,axis=0)
        meanEE_namaster=np.mean(list_EE_namaster,axis=0)
        meanBB_namaster=np.mean(list_BB_namaster,axis=0)

l,ps_theory=pspy_utils.ps_lensed_theory_to_dict(clfile,'Cl',lmax=lmax)
ps_theory_b=so_mcm.apply_Bbl(Bbl,ps_theory,spectra=spectra)
ps_theory_b_pure=so_mcm.apply_Bbl(Bbl_pure,ps_theory,spectra=spectra)


l,ps_theory=pspy_utils.ps_lensed_theory_to_dict(clfile,'Cl',lmax=3*nside)
ps_theory_b_namaster_pure=w_yp.decouple_cell(w_yp.couple_cell([ps_theory['EE'],ps_theory['BB']*0,ps_theory['BB']*0,ps_theory['BB']]))

#plt.errorbar(lb-5,ps_theory_b*lb**2/(2*np.pi),label='namaster',color='orange')
plt.errorbar(leff-5,meanBB_namaster*leff**2/(2*np.pi),std_BB_namaster*leff**2/(2*np.pi),fmt='.',label='namaster',color='orange')
plt.errorbar(leff-5,meanBB_pure_namaster*leff**2/(2*np.pi),std_BB_pure_namaster*leff**2/(2*np.pi),fmt='.',label='namaster pure',color='purple')
plt.errorbar(lb+5,meanBB*lb**2/(2*np.pi),std_BB*lb**2/(2*np.pi),fmt='.',label='pspy',color='red')
plt.errorbar(lb+5,meanBB_pure*lb**2/(2*np.pi),std_BB_pure*lb**2/(2*np.pi),fmt='.',label='pspy pure',color='blue')
plt.plot(l,ps_theory['BB'],label='theory',color='black')
plt.legend()
plt.show()

plt.errorbar(leff,meanBB_pure_namaster*leff**2/(2*np.pi),std_BB_pure_namaster*leff**2/(2*np.pi),fmt='.',label='namaster pure',color='purple')
plt.plot(leff,ps_theory_b_namaster_pure[3]*leff**2/(2*np.pi))
plt.show()


plt.errorbar(lb,meanBB_pure*lb**2/(2*np.pi),std_BB_pure*lb**2/(2*np.pi),fmt='.',label='pspy pure',color='purple')
plt.plot(lb,ps_theory_b_pure['BB']*lb**2/(2*np.pi))
plt.show()

