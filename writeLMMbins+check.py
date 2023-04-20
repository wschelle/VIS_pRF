#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:06:24 2023

@author: WauWter
"""

import os
os.chdir('/home/wouter')
import numpy as np
import matplotlib.pyplot as plt
from Python.python_scripts.wauwternifti import readnii
from Python.python_scripts.wauwterfmri import *
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

rdir='/Fridge/users/wouter/Students/Glenn/subjects/'
subs=['sub-visual01','sub-visual02','sub-visual03','sub-visual05','sub-visual06',
      'sub-visual08','sub-visual09','sub-visual10','sub-visual11','sub-visual12']

nsub=len(subs)
scandur=850.
rt=0.1
maxecc=3.5
necc=np.ceil(maxecc*2)
nvar=3.

laylow=2.
layhigh=19.
maxlay=(layhigh+1)-laylow
nlay=6
laynorm=maxlay/nlay

varname=['V1','V2','V3']
tits=['subid','pol','ecc','pRF','ampl','rsq','mod','var','lay']

# GE
seqs=['GE']
nseq=len(seqs)
if (seqs[0] == 'GE') | (seqs[0] == 'SE'):
    fieldst=['ses-7T']
else:
    fieldst=['ses-3T']

sess=[""]
for i in range(nseq):
    sess[i]=fieldst[i]+seqs[i]

# subs=['sub-visual02','sub-visual03','sub-visual05','sub-visual06',$
#   'sub-visual08','sub-visual11','sub-visual12']
    
# subs=['sub-visual12']

if seqs[0] == 'GE':
    fx=128
    fy=128
    fz=7
elif seqs[0] == 'SE':
    fx=128
    fy=128
    fz=5
elif seqs[0] == 'MB':
    fx=96
    fy=96
    fz=30

polx=np.zeros([nsub,fx,fy,fz])
eccx=np.zeros([nsub,fx,fy,fz])
prfx=np.zeros([nsub,fx,fy,fz])
rsqx=np.zeros([nsub,fx,fy,fz])
ampx=np.zeros([nsub,fx,fy,fz])
fvax=np.zeros([nsub,fx,fy,fz])
j=0
for i in range(nsub):
    pol,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-pol.nii')
    ecc,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-ecc.nii')
    prf,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-prf.nii')
    rsq,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-rsq.nii')
    amp,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-beta.nii')
    fval,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-fval.nii')
    polx[i,:,:,:]=pol
    eccx[i,:,:,:]=ecc
    prfx[i,:,:,:]=prf
    rsqx[i,:,:,:]=rsq
    ampx[i,:,:,:]=amp
    fvax[i,:,:,:]=fval
    del pol,ecc,prf,rsq,amp,fval
    

nvox=fx*fy*fz
polx=np.reshape(polx,[nsub,nvox])
eccx=np.reshape(eccx,[nsub,nvox])
prfx=np.reshape(prfx,[nsub,nvox])
ampx=np.reshape(ampx,[nsub,nvox])
rsqx=np.reshape(rsqx,[nsub,nvox])
rsqx[np.where(np.isnan(rsqx))]=0
fvax=np.reshape(fvax,[nsub,nvox])

dfn,dfd=4,494
#False-Discovery-Rate correction
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',cutoff_only=True)
# print(tc)
#Holm-Bonferroni correction
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='holm',cutoff_only=True)
# print(tc)
#Bonferroni correction
tc=np.zeros(nsub)
for i in range(nsub):
    tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='bonferroni',cutoff_only=True)
print(tc)

# rsm=np.sqrt(rsqx)
# tsm=rval2tval(rsm,dfd)
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn=494,test='t',tail=2,method='bonferroni',cutoff_only=True)
# print(tc)

lay=np.zeros([nsub,fx,fy,fz])
var=np.zeros([nsub,fx,fy,fz])
for i in range(nsub):
    if seqs[0]=='MB':
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/laySE.nii'
    else:
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/lay.nii'
    laynii,hdr=readnii(layfile,scaling=False)
    varnii,hdr=readnii(rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/varea.nii',scaling=False)
    lay[i,:,:,:]=laynii
    var[i,:,:,:]=varnii
    del laynii,varnii

var[var > nvar]=0

# old layer interpolation
# lay[where(lay eq 1)]++
# lay[where(lay eq 20)]--
# lay[where(lay ne 0)]--
# lay/=laynorm
# lay=ceil(lay)

# new layer interpolation: throw away K bottom ones: too much WM
lay[lay < laylow]=0
lay[lay > layhigh]=0
lay[lay != 0]-=(laylow-1)
lay/=laynorm
lay=np.ceil(lay)

var=np.reshape(var,[nsub,nvox])
lay=np.reshape(lay,[nsub,nvox])

lmm=np.zeros(9)
lmmtits0=['sid','seq','pol','ecc','prf','amp','rsq','var','lay']
for i in tqdm(range(nsub)):
    if np.sum((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))>1:
        tmp=np.where((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))[0]
        s=np.zeros(len(tmp))
        s+=(i+1)
        m=np.zeros(len(tmp))
        m+=1 #GE
        p=polx[i,tmp]
        e=eccx[i,tmp]
        f=prfx[i,tmp]
        a=ampx[i,tmp]
        q=rsqx[i,tmp]
        v=var[i,tmp]
        l=lay[i,tmp]
        sm0=np.concatenate((s,m,p,e,f,a,q,v,l),axis=0)
        lmm=np.vstack((lmm,np.reshape(sm0,[9,len(tmp)]).T))

lmmge=pd.DataFrame(lmm[1:,:],columns=lmmtits0)

lmm=lmm[1:,:].astype(np.float64)

lmmtits="sid\tseq\tpol\tecc\tprf\tamp\trsq\tvar\tlay"
fmt = "\t".join(["%s"] + ["%10.6e"] * (lmm.shape[1]-1))
np.savetxt(rdir+'Group/LMM_GEpy.tsv', lmm, fmt=fmt, header=lmmtits, delimiter ='\t')


nvar=int(nvar)
necc=int(necc)
nlay=int(nlay)
mprf=np.zeros([nsub,nvar,necc,nlay])
mamp=np.zeros([nsub,nvar,necc,nlay])
for i in tqdm(range(nsub)):
    for j in range(nvar):
        for k in range(necc):
            for l in range(nlay):
                if np.sum(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])>0:
                    mprf[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])
                    mamp[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),5])
                else:
                    mprf[i,j,k,l]=np.nan
                    mamp[i,j,k,l]=np.nan


prf_varecc1=np.nanmean(mprf,axis=3)
amp_varecc1=np.nanmean(mamp,axis=3)

prf_varlay1=np.nanmean(mprf,axis=2)
amp_varlay1=np.nanmean(mamp,axis=2)

prf_ve_p=np.mean(prf_varecc1,axis=0)
prf_ve_e=np.std(prf_varecc1,axis=0)/np.sqrt(nsub)
amp_ve_p=np.mean(amp_varecc1,axis=0)
amp_ve_e=np.std(amp_varecc1,axis=0)/np.sqrt(nsub)

prf_vl_p=np.mean(prf_varlay1,axis=0)
prf_vl_e=np.std(prf_varlay1,axis=0)/np.sqrt(nsub)
amp_vl_p=np.mean(amp_varlay1,axis=0)
amp_vl_e=np.std(amp_varlay1,axis=0)/np.sqrt(nsub)


xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_ve_p[i,:] - prf_ve_e[i,:], prf_ve_p[i,:] + prf_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_ve_p.min()-0.2,ymax=prf_ve_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_ve_p[i,:] - amp_ve_e[i,:], amp_ve_p[i,:] + amp_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_ve_p.min()-0.2,ymax=amp_ve_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()


xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_vl_p[i,:] - prf_vl_e[i,:], prf_vl_p[i,:] + prf_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_vl_p.min()-0.2,ymax=prf_vl_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_vl_p[i,:] - amp_vl_e[i,:], amp_vl_p[i,:] + amp_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_vl_p.min()-0.2,ymax=amp_vl_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()


md = smf.mixedlm("prf ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())
md = smf.mixedlm("amp ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())



# SE
seqs=['SE']
nseq=len(seqs)
if (seqs[0] == 'GE') | (seqs[0] == 'SE'):
    fieldst=['ses-7T']
else:
    fieldst=['ses-3T']

sess=[""]
for i in range(nseq):
    sess[i]=fieldst[i]+seqs[i]

if seqs[0] == 'GE':
    fx=128
    fy=128
    fz=7
elif seqs[0] == 'SE':
    fx=128
    fy=128
    fz=5
elif seqs[0] == 'MB':
    fx=96
    fy=96
    fz=30

polx=np.zeros([nsub,fx,fy,fz])
eccx=np.zeros([nsub,fx,fy,fz])
prfx=np.zeros([nsub,fx,fy,fz])
rsqx=np.zeros([nsub,fx,fy,fz])
ampx=np.zeros([nsub,fx,fy,fz])
fvax=np.zeros([nsub,fx,fy,fz])
j=0
for i in range(nsub):
    pol,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-pol.nii')
    ecc,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-ecc.nii')
    prf,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-prf.nii')
    rsq,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-rsq.nii')
    amp,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-beta.nii')
    fval,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-fval.nii')
    polx[i,:,:,:]=pol
    eccx[i,:,:,:]=ecc
    prfx[i,:,:,:]=prf
    rsqx[i,:,:,:]=rsq
    ampx[i,:,:,:]=amp
    fvax[i,:,:,:]=fval
    del pol,ecc,prf,rsq,amp,fval
    

nvox=fx*fy*fz
polx=np.reshape(polx,[nsub,nvox])
eccx=np.reshape(eccx,[nsub,nvox])
prfx=np.reshape(prfx,[nsub,nvox])
ampx=np.reshape(ampx,[nsub,nvox])
rsqx=np.reshape(rsqx,[nsub,nvox])
rsqx[np.where(np.isnan(rsqx))]=0
fvax=np.reshape(fvax,[nsub,nvox])

dfn,dfd=4,494
#False-Discovery-Rate correction
tc=np.zeros(nsub)
for i in range(nsub):
    tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',cutoff_only=True)
print(tc)
#Holm-Bonferroni correction
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='holm',cutoff_only=True)
# print(tc)
# #Bonferroni correction
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='bonferroni',cutoff_only=True)
# print(tc)

# rsm=np.sqrt(rsqx)
# tsm=rval2tval(rsm,dfd)
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn=494,test='t',tail=2,method='bonferroni',cutoff_only=True)
# print(tc)

lay=np.zeros([nsub,fx,fy,fz])
var=np.zeros([nsub,fx,fy,fz])
for i in range(nsub):
    if seqs[0]=='MB':
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/laySE.nii'
    else:
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/lay.nii'
    laynii,hdr=readnii(layfile,scaling=False)
    varnii,hdr=readnii(rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/varea.nii',scaling=False)
    lay[i,:,:,:]=laynii
    var[i,:,:,:]=varnii
    del laynii,varnii

var[var > nvar]=0

# old layer interpolation
# lay[where(lay eq 1)]++
# lay[where(lay eq 20)]--
# lay[where(lay ne 0)]--
# lay/=laynorm
# lay=ceil(lay)

# new layer interpolation: throw away K bottom ones: too much WM
lay[lay < laylow]=0
lay[lay > layhigh]=0
lay[lay != 0]-=(laylow-1)
lay/=laynorm
lay=np.ceil(lay)

var=np.reshape(var,[nsub,nvox])
lay=np.reshape(lay,[nsub,nvox])

lmm=np.zeros(9)
lmmtits0=['sid','seq','pol','ecc','prf','amp','rsq','var','lay']
for i in tqdm(range(nsub)):
    if np.sum((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))>1:
        tmp=np.where((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))[0]
        s=np.zeros(len(tmp))
        s+=(i+1)
        m=np.zeros(len(tmp))
        m+=2 #SE
        p=polx[i,tmp]
        e=eccx[i,tmp]
        f=prfx[i,tmp]
        a=ampx[i,tmp]
        q=rsqx[i,tmp]
        v=var[i,tmp]
        l=lay[i,tmp]
        sm0=np.concatenate((s,m,p,e,f,a,q,v,l),axis=0)
        lmm=np.vstack((lmm,np.reshape(sm0,[9,len(tmp)]).T))

lmmse=pd.DataFrame(lmm[1:,:],columns=lmmtits0)

lmm=lmm[1:,:].astype(np.float64)

lmmtits="sid\tseq\tpol\tecc\tprf\tamp\trsq\tvar\tlay"
fmt = "\t".join(["%s"] + ["%10.6e"] * (lmm.shape[1]-1))
np.savetxt(rdir+'Group/LMM_SEpy.tsv', lmm, fmt=fmt, header=lmmtits, delimiter ='\t')


nvar=int(nvar)
necc=int(necc)
nlay=int(nlay)
mprf=np.zeros([nsub,nvar,necc,nlay])
mamp=np.zeros([nsub,nvar,necc,nlay])
for i in tqdm(range(nsub)):
    for j in range(nvar):
        for k in range(necc):
            for l in range(nlay):
                if np.sum(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])>0:
                    mprf[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])
                    mamp[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),5])
                else:
                    mprf[i,j,k,l]=np.nan
                    mamp[i,j,k,l]=np.nan


prf_varecc2=np.nanmean(mprf,axis=3)
amp_varecc2=np.nanmean(mamp,axis=3)

prf_varlay2=np.nanmean(mprf,axis=2)
amp_varlay2=np.nanmean(mamp,axis=2)

prf_ve_p=np.nanmean(prf_varecc2,axis=0)
prf_ve_e=np.nanstd(prf_varecc2,axis=0)/np.sqrt(nsub)
amp_ve_p=np.nanmean(amp_varecc2,axis=0)
amp_ve_e=np.nanstd(amp_varecc2,axis=0)/np.sqrt(nsub)

prf_vl_p=np.nanmean(prf_varlay2,axis=0)
prf_vl_e=np.nanstd(prf_varlay2,axis=0)/np.sqrt(nsub)
amp_vl_p=np.nanmean(amp_varlay2,axis=0)
amp_vl_e=np.nanstd(amp_varlay2,axis=0)/np.sqrt(nsub)


xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_ve_p[i,:] - prf_ve_e[i,:], prf_ve_p[i,:] + prf_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_ve_p.min()-0.2,ymax=prf_ve_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_ve_p[i,:] - amp_ve_e[i,:], amp_ve_p[i,:] + amp_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_ve_p.min()-0.2,ymax=amp_ve_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()


xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_vl_p[i,:] - prf_vl_e[i,:], prf_vl_p[i,:] + prf_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_vl_p.min()-0.2,ymax=prf_vl_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_vl_p[i,:] - amp_vl_e[i,:], amp_vl_p[i,:] + amp_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_vl_p.min()-0.2,ymax=amp_vl_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()

md = smf.mixedlm("prf ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())
md = smf.mixedlm("amp ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())


# MB
seqs=['MB']
nseq=len(seqs)
if (seqs[0] == 'GE') | (seqs[0] == 'SE'):
    fieldst=['ses-7T']
else:
    fieldst=['ses-3T']

sess=[""]
for i in range(nseq):
    sess[i]=fieldst[i]+seqs[i]

if seqs[0] == 'GE':
    fx=128
    fy=128
    fz=7
elif seqs[0] == 'SE':
    fx=128
    fy=128
    fz=5
elif seqs[0] == 'MB':
    fx=96
    fy=96
    fz=30

polx=np.zeros([nsub,fx,fy,fz])
eccx=np.zeros([nsub,fx,fy,fz])
prfx=np.zeros([nsub,fx,fy,fz])
rsqx=np.zeros([nsub,fx,fy,fz])
ampx=np.zeros([nsub,fx,fy,fz])
fvax=np.zeros([nsub,fx,fy,fz])
j=0
for i in range(nsub):
    pol,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-pol.nii')
    ecc,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-ecc.nii')
    prf,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-prf.nii')
    rsq,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-rsq.nii')
    amp,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-beta.nii')
    fval,hdr=readnii(rdir+subs[i]+'/'+sess[j]+'/derivatives/nii/'+subs[i]+'-NORDIC_'+sess[j]+'-circprf-shrf-fval.nii')
    polx[i,:,:,:]=pol
    eccx[i,:,:,:]=ecc
    prfx[i,:,:,:]=prf
    rsqx[i,:,:,:]=rsq
    ampx[i,:,:,:]=amp
    fvax[i,:,:,:]=fval
    del pol,ecc,prf,rsq,amp,fval
    

nvox=fx*fy*fz
polx=np.reshape(polx,[nsub,nvox])
eccx=np.reshape(eccx,[nsub,nvox])
prfx=np.reshape(prfx,[nsub,nvox])
ampx=np.reshape(ampx,[nsub,nvox])
rsqx=np.reshape(rsqx,[nsub,nvox])
rsqx[np.where(np.isnan(rsqx))]=0
fvax=np.reshape(fvax,[nsub,nvox])

dfn,dfd=4,494
#False-Discovery-Rate correction
tc=np.zeros(nsub)
for i in range(nsub):
    tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',cutoff_only=True)
print(tc)
#Holm-Bonferroni correction
tc=np.zeros(nsub)
for i in range(nsub):
    tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='holm',cutoff_only=True)
print(tc)
#Bonferroni correction
tc=np.zeros(nsub)
for i in range(nsub):
    tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn,dfd,test='f',method='bonferroni',cutoff_only=True)
print(tc)

# rsm=np.sqrt(rsqx)
# tsm=rval2tval(rsm,dfd)
# tc=np.zeros(nsub)
# for i in range(nsub):
#     tc[i]=multiple_comparison(fvax[i,rsqx[i,:]!=0],dfn=494,test='t',tail=2,method='bonferroni',cutoff_only=True)
# print(tc)

lay=np.zeros([nsub,fx,fy,fz])
var=np.zeros([nsub,fx,fy,fz])
for i in range(nsub):
    if seqs[0]=='MB':
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/laySE.nii'
    else:
        layfile=rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/lay.nii'
    laynii,hdr=readnii(layfile,scaling=False)
    varnii,hdr=readnii(rdir+subs[i]+'/'+sess[0]+'/func/'+seqs[0]+'_PRF_1/varea.nii',scaling=False)
    lay[i,:,:,:]=laynii
    var[i,:,:,:]=varnii
    del laynii,varnii

var[var > nvar]=0

# old layer interpolation
# lay[where(lay eq 1)]++
# lay[where(lay eq 20)]--
# lay[where(lay ne 0)]--
# lay/=laynorm
# lay=ceil(lay)

# new layer interpolation: throw away K bottom ones: too much WM
lay[lay < laylow]=0
lay[lay > layhigh]=0
lay[lay != 0]-=(laylow-1)
lay/=laynorm
lay=np.ceil(lay)

var=np.reshape(var,[nsub,nvox])
lay=np.reshape(lay,[nsub,nvox])

lmm=np.zeros(9)
lmmtits0=['sid','seq','pol','ecc','prf','amp','rsq','var','lay']
for i in tqdm(range(nsub)):
    if np.sum((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))>1:
        tmp=np.where((fvax[i,:]>tc[i]) & (var[i,:]>0) & (lay[i,:]>0) & (eccx[i,:]<maxecc))[0]
        s=np.zeros(len(tmp))
        s+=(i+1)
        m=np.zeros(len(tmp))
        m+=3 #MB
        p=polx[i,tmp]
        e=eccx[i,tmp]
        f=prfx[i,tmp]
        a=ampx[i,tmp]
        q=rsqx[i,tmp]
        v=var[i,tmp]
        l=lay[i,tmp]
        sm0=np.concatenate((s,m,p,e,f,a,q,v,l),axis=0)
        lmm=np.vstack((lmm,np.reshape(sm0,[9,len(tmp)]).T))

lmmmb=pd.DataFrame(lmm[1:,:],columns=lmmtits0)

lmm=lmm[1:,:].astype(np.float64)


lmmtits="sid\tseq\tpol\tecc\tprf\tamp\trsq\tvar\tlay"
fmt = "\t".join(["%s"] + ["%10.6e"] * (lmm.shape[1]-1))
np.savetxt(rdir+'Group/LMM_MBpy.tsv', lmm, fmt=fmt, header=lmmtits, delimiter ='\t')


nvar=int(nvar)
necc=int(necc)
nlay=int(nlay)
mprf=np.zeros([nsub,nvar,necc,nlay])
mamp=np.zeros([nsub,nvar,necc,nlay])
for i in tqdm(range(nsub)):
    for j in range(nvar):
        for k in range(necc):
            for l in range(nlay):
                if np.sum(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])>0:
                    mprf[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),4])
                    mamp[i,j,k,l]=np.mean(lmm[(lmm[:,0]==i+1) & (lmm[:,3]>k/2) & (lmm[:,3]<(k+1)/2)
                                          & (lmm[:,7]==j+1) & (lmm[:,8]==l+1),5])


prf_varecc3=np.mean(mprf,axis=3)
amp_varecc3=np.mean(mamp,axis=3)

prf_varlay3=np.mean(mprf,axis=2)
amp_varlay3=np.mean(mamp,axis=2)

prf_ve_p=np.mean(prf_varecc3,axis=0)
prf_ve_e=np.std(prf_varecc3,axis=0)/np.sqrt(nsub)
amp_ve_p=np.mean(amp_varecc3,axis=0)
amp_ve_e=np.std(amp_varecc3,axis=0)/np.sqrt(nsub)

prf_vl_p=np.mean(prf_varlay3,axis=0)
prf_vl_e=np.std(prf_varlay3,axis=0)/np.sqrt(nsub)
amp_vl_p=np.mean(amp_varlay3,axis=0)
amp_vl_e=np.std(amp_varlay3,axis=0)/np.sqrt(nsub)


xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_ve_p[i,:] - prf_ve_e[i,:], prf_ve_p[i,:] + prf_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_ve_p.min()-0.2,ymax=prf_ve_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,maxecc,0.5)+0.5
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_ve_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_ve_p[i,:] - amp_ve_e[i,:], amp_ve_p[i,:] + amp_ve_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_ve_p.min()-0.2,ymax=amp_ve_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('Eccentricity (dva)')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()


xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,prf_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, prf_vl_p[i,:] - prf_vl_e[i,:], prf_vl_p[i,:] + prf_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=prf_vl_p.min()-0.2,ymax=prf_vl_p.max()+0.2)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('pRF size')
plt.show()

xax=np.arange(0,nlay)+1
fig, axes = plt.subplots(1, nvar, figsize=(10,4))
for i in range(nvar):
    axes[i].plot(xax,amp_vl_p[i,:],'o',color='navy')
    axes[i].fill_between(xax, amp_vl_p[i,:] - amp_vl_e[i,:], amp_vl_p[i,:] + amp_vl_e[i,:], alpha=0.2)
    axes[i].set_ylim(ymin=amp_vl_p.min()-0.2,ymax=amp_vl_p.max()+0.5)
    axes[i].set_title(varname[i])
    if i==nvar//2:
        axes[i].set_xlabel('WM <---> GM')
    if i==0:
        axes[i].set_ylabel('% signal change')
plt.show()


md = smf.mixedlm("prf ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())
md = smf.mixedlm("amp ~ var * lay + ecc", data=lmmge, groups="sid")
mdf = md.fit()
print(mdf.summary())



#statsmodels works, but DoF are enormous
lmm7t=pd.concat([lmmge,lmmse])
lmm7t.to_csv(rdir+'Group/LMM_GESEpy.tsv',sep="\t")

md = smf.mixedlm("prf ~ (ecc + var + lay) * seq ", data=lmm7t, groups="sid")
mdf = md.fit()
print(mdf.summary())
A=np.identity(len(mdf.params))
A=A[1:,:]
print(mdf.f_test(A))
print(mdf.wald_test(A,scalar=True))

md = smf.mixedlm("amp ~ (ecc + var + lay) * seq ", data=lmm7t, groups="sid")
mdf = md.fit()
print(mdf.summary())
A=np.identity(len(mdf.params))
A=A[1:,:]
print(mdf.f_test(A))
print(mdf.wald_test(A,scalar=True))


lmmall=pd.concat([lmmge,lmmse,lmmmb])
lmmall.to_csv(rdir+'Group/LMM_GESEMBpy.tsv',sep="\t")

md = smf.mixedlm("prf ~ (ecc + var) * seq", data=lmmall, groups="sid")
mdf = md.fit()
print(mdf.summary())
A=np.identity(len(mdf.params))
A=A[1:,:]
print(mdf.f_test(A))
print(mdf.wald_test(A,scalar=True))


md = smf.mixedlm("amp ~ (ecc + var) * seq", data=lmmall, groups="sid")
mdf = md.fit()
print(mdf.summary())
A=np.identity(len(mdf.params))
A=A[1:,:]
print(mdf.f_test(A))
print(mdf.wald_test(A,scalar=True))

