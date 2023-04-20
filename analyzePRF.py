#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:49:26 2023

@author: WauWter
"""
#If run from terminal, do this in terminal first:
# export PYTHONPATH="$PYTHONPATH:/home/wouter/"
import os
os.chdir('/home/wouter/')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Python.python_scripts.wauwterfmri import *
from Python.python_scripts.wauwternifti import readnii,savenii
from Python.python_scripts.wauwtermisc import rebin
from scipy.signal import savgol_filter
from tqdm import tqdm
import copy
import lmfit
from multiprocessing import Pool

rdir='/Fridge/users/wouter/Students/Glenn/subjects/'
seq='GE'
ses='ses-7T'+seq
tasks=[seq+'_PRF_1',seq+'_PRF_2']
ntask=len(tasks)
vxthres=100
cut=0.0125
fmresize=6

subs=['sub-visual01','sub-visual02','sub-visual03','sub-visual05','sub-visual06',
'sub-visual08','sub-visual09','sub-visual10','sub-visual11','sub-visual12']
nsubs=len(subs)
scandur=850
TR=scandur/1000

filename = '/home/wouter/Fridge/Students/Glenn/subjects/fmatrix.h5'
f = h5py.File(filename, 'r')
fmatrix = f['fmatrix']
fmatrix=np.asarray(fmatrix,dtype=np.float32)
f.close()

sx=fmatrix.shape[0]
sy=fmatrix.shape[1]
sxf=sx/fmresize

ii=9
if os.path.exists(rdir+subs[ii]+'/'+ses+'/func/'+tasks[0]+'/'):

    sub=subs[ii]
    sdir=rdir+sub+'/'+ses+'/'
    task1=tasks[0]
    task2=tasks[1]
    tdir1=sdir+'func/'+task1+'/'
    nifti1=tdir1+'NORDIC_'+task1+'-k-mc-w.nii'
    tdir2=sdir+'func/'+task2+'/'
    nifti2=tdir2+'NORDIC_'+task2+'-k-mc-w.nii'
    ddir=sdir+'derivatives/'
    pddir=ddir+'pics/'
    nddir=ddir+'nii/'
    sddir=ddir+'sav/'
 

    # Read nifti
    x1,hdr1=readnii(nifti1)
    x2,hdr2=readnii(nifti2)
    
    fx=x1.shape[0]
    fy=x1.shape[1]
    fz=x1.shape[2]
    nscans1=x1.shape[3]
    nscans=nscans1*2
    nvox=fx*fy*fz
    
    sm1=np.reshape(x1,(nvox,nscans1))
    sm2=np.reshape(x2,(nvox,nscans1))
    del x1,x2
    sm=np.concatenate((sm1,sm2),axis=1)

    #make simple mask
    mask1=np.zeros(nvox,dtype=np.int16)
    mask1[np.mean(sm1,axis=1) > vxthres]=1
    
    mask2=np.zeros(nvox,dtype=np.int16)
    mask2[np.mean(sm2,axis=1) > vxthres]=1
    
    mask=mask1*mask2
    del sm1, sm2, mask1, mask2
    
    taskreg=np.zeros([1,nscans],dtype=np.float32)
    taskreg[0,0:nscans1]=1
    
    print('high-pass filtering:')
    sm2=hpfilt(sm,TR,cut,addfilt=taskreg,mask=mask,convperc=1,showfiltmat=False)
    del taskreg
    
    sm3=np.zeros(sm2.shape)
    print('temporal smoothing:')
    for i in tqdm(range(nvox)):
        if mask[i]!=0:
            sm3[i,:]=savgol_filter(sm2[i,:], 20, 4, mode='mirror')
    del sm,sm2
            
    fm=copy.deepcopy(fmatrix[:,:,0:nscans1])
    fm=np.concatenate((fm,fm),axis=2)
    fmr=np.zeros([fmresize,fmresize,nscans],dtype=np.float32)
    for i in range(nscans):
        fmr[:,:,i]=rebin(np.squeeze(fm[:,:,i]),(fmresize,fmresize))
    del fmatrix
    
    print('calculating coarse correlations:')
    cmat=np.zeros([nvox,fmresize,fmresize])
    for i in tqdm(range(nvox)):
        if mask[i]!=0:
            for j in range(fmresize):
                for k in range(fmresize):
                    tmp=np.corrcoef(np.squeeze(fmr[j,k,:]),np.squeeze(sm3[i,:]))
                    cmat[i,j,k]=tmp[0,1]
    
    gest=np.zeros([nvox,6],dtype=np.float32)
    cmat2=np.zeros(nvox,dtype=np.float32)
    halfxy=(sx-1)/2
    print('calculating pRF estimates)')
    for i in tqdm(range(nvox)):
        if mask[i]!=0:
            gest[i,0]=np.mean(sm3[i,:])
            gest[i,1]=np.std(sm3[i,:])
            topfac=np.where(cmat[i,:,:]==np.max(cmat[i,:,:]))
            gest[i,2]=topfac[0][0]*sxf+(sxf/2)
            gest[i,3]=topfac[1][0]*sxf+(sxf/2)
            tmpecc=np.sqrt((gest[i,2]-halfxy)**2 + (gest[i,3]-halfxy)**2)
            gest[i,4]=tmpecc/3.
            gest[i,5]=1.
            cmat2[i]=cmat[i,topfac[0][0],topfac[1][0]]
            
    gest[(mask!=0)&(gest[:,4]==0),4]=1
    tmat=rval2tval(cmat2,nscans-1)
    del cmat, cmat2
    
    mp_zfit=np.zeros(7,dtype=np.float32)
    m2=np.zeros(nvox,dtype=np.int16)
    m2[tmat>=1]=1
    del tmat
    
    xgrid = np.arange(sx,dtype=np.float32)
    ygrid = np.arange(sx,dtype=np.float32)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    
    m2list=np.where(m2==1)[0]
    mp_zfit=np.zeros(7,dtype=np.float32)
    sm4=sm3[m2list,:]
    gest2=gest[m2list,:]

    print('Precise calculation pRF:')
    def mp_prf2d(voxel):
        params = lmfit.Parameters()
        params.add('cons', gest2[voxel,0])
        params.add('amp', gest2[voxel,1],min=0)
        params.add('centerX', gest2[voxel,2],min=-0.5,max=sx-0.5)
        params.add('centerY', gest2[voxel,3],min=-0.5,max=sy-0.5)
        params.add('XYsigma', gest2[voxel,4],min=0.5,max=2*sx)
        params.add('XYsigmaRatio', gest2[voxel,5],vary=False)
        zfit = lmfit.minimize(lmfit_prf2d, params, args=(fm, sm4[voxel,:], xgrid, ygrid))
        mp_zfit[0]=zfit.params['cons'].value
        mp_zfit[1]=zfit.params['amp'].value
        mp_zfit[2]=zfit.params['centerX'].value
        mp_zfit[3]=zfit.params['centerY'].value
        mp_zfit[4]=zfit.params['XYsigma'].value
        mp_zfit[5]=zfit.params['XYsigmaRatio'].value
        mp_zfit[6]=zfit.chisqr
        return mp_zfit

    p=Pool(20)
    mp_result=p.map(mp_prf2d, range(len(m2list)))
    p.close()
    
    prf_zfit=np.zeros([nvox,7],dtype=np.float32)
    for i in range(len(m2list)):
        prf_zfit[m2list[i],:]=mp_result[i]    
    del mp_result,mp_zfit
    
    # sm4=sm3[m2==1,:]
    # m2list=np.where(m2>0)[0]
    # m2vox=len(m2list)
    # prf_zfit=np.zeros([nvox,7],dtype=np.float32)
    
    # print('Precise calculation pRF:')
    # for i in tqdm(range(m2vox)):
    #     params = lmfit.Parameters()
    #     params.add('cons', gest[m2list[i],0])
    #     params.add('amp', gest[m2list[i],1],min=0)
    #     params.add('centerX', gest[m2list[i],2],min=-0.5,max=sx-0.5)
    #     params.add('centerY', gest[m2list[i],3],min=-0.5,max=sy-0.5)
    #     params.add('XYsigma', gest[m2list[i],4],min=0.5,max=2*sx)
    #     params.add('XYsigmaRatio', gest[m2list[i],5],vary=False)
    #     zfit = lmfit.minimize(lmfit_prf2d, params, args=(fm, sm4[i,:], xgrid, ygrid))
    #     prf_zfit[m2list[i],0]=zfit.params['cons'].value
    #     prf_zfit[m2list[i],1]=zfit.params['amp'].value
    #     prf_zfit[m2list[i],2]=zfit.params['centerX'].value
    #     prf_zfit[m2list[i],3]=zfit.params['centerY'].value
    #     prf_zfit[m2list[i],4]=zfit.params['XYsigma'].value
    #     prf_zfit[m2list[i],5]=zfit.params['XYsigmaRatio'].value
    #     prf_zfit[m2list[i],6]=zfit.chisqr
    
    npar=5
    dfn=(npar-1)-1
    dfd=nscans-(npar-1)-1
    prf_yfit=np.zeros(sm3.shape,dtype=np.float32)
    print('determine pRF model:')
    for i in tqdm(range(nvox)):
        if mask[i]!=0:
            prf_yfit[i,:]=prf2d(fm,np.squeeze(prf_zfit[i,0:6]),xgrid,ygrid)
    
    del xgrid, ygrid

    print('calculate f-statistic:')
    fval=np.zeros(nvox,dtype=np.float32)
    rval=np.zeros(nvox,dtype=np.float32)
    for i in tqdm(range(nvox)):
        if prf_zfit[i,6] > 0:
            fval[i]=(np.sum((np.mean(sm3[i,:])-prf_yfit[i,:])**2)/dfn) - (prf_zfit[i,6]/dfd)
            tmp=np.corrcoef(np.squeeze(prf_yfit[i,:]),np.squeeze(sm3[i,:]))
            rval[i]=tmp[0,1]
    

    degvis=(12/sx)
    prf_zfit[:,4]*=degvis

    center=copy.deepcopy(prf_zfit[:,2:4])
    center[fval>0,0]-=(sx-1)/2.
    center[fval>0,1]-=(sy-1)/2.
    center*=degvis
    ecc=np.zeros(nvox,dtype=np.float32)
    print('calculate eccentricity:')
    for i in tqdm(range(nvox)):
        if fval[i] > 0:
            ecc[i]=np.sqrt(center[i,0]**2+center[i,1]**2)
    print('calculate polar angle:') 
    pol=np.zeros(nvox,dtype=np.float32)
    for i in tqdm(range(nvox)):
        if fval[i]>0:
            if (center[i,1]>=0) & (ecc[i] != 0):
                pol[i]=np.arccos(center[i,0]/ecc[i])
            elif center[i,1]<0:
                pol[i]=-np.arccos(center[i,0]/ecc[i])
            pol[i]*=(180/np.pi)

    pol[fval > 0]+=270
    pol[(fval > 0) & (pol > 360)]-=360
    prfsize=copy.deepcopy(prf_zfit[:,4])

    pol2=np.reshape(pol,(fx,fy,fz))
    ecc2=np.reshape(ecc,(fx,fy,fz))
    prf2=np.reshape(prfsize,(fx,fy,fz))
    rsq2=np.reshape(rval**2,(fx,fy,fz))
    amp2=np.reshape(prf_zfit[:,1],(fx,fy,fz))
    fval2=np.reshape(fval,(fx,fy,fz))
    
    hdr1['vox_offset']=352
    hdr1['dim']=(3,fx,fy,fz,1,1,1,1)
    hdr1['pixdim']=(-1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    hdr1['scl_slope']=1


    savenii(pol2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-pol-py.nii.gz')
    savenii(ecc2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-ecc-py.nii.gz')
    savenii(prf2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-prf-py.nii.gz')
    savenii(rsq2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-rsq-py.nii.gz')
    savenii(amp2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-beta-py.nii.gz')
    savenii(fval2,hdr1,nddir+sub+'-NORDIC_'+ses+'-circprf-fval-py.nii.gz')
    
    
    
    


    
    
    
    
