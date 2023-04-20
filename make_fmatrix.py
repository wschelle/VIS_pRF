#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 00:47:50 2023

@author: wouter
"""
import os
os.chdir('/home/wouter')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from Python.python_scripts.wauwterfmri import *
from scipy.interpolate import interp1d

filename = '/home/wouter/Fridge/Students/Glenn/subjects/prfstim48x48w.h5'
f = h5py.File(filename, 'r')
dset = f['prfstim']
dset=np.asarray(dset,dtype=np.int16)
plt.imshow(dset[:,:,15])
plt.show()
plt.imshow(dset[:,:,35])
plt.show()
plt.imshow(dset[:,:,71])
plt.show()

ons=np.loadtxt('/home/wouter/Fridge/Students/Glenn/subjects/sub-visual01/ses-7TGE/log/sub-visual01GE_ses-umc7t01_task-prf_run-1.tsv',
               skiprows=1,usecols=(0,1,2),delimiter='\t')

nscans=250
TR=0.85
nt=int(np.round(nscans*TR))

nx=dset.shape[0]
ny=dset.shape[1]
ne=dset.shape[2]

upsample_factor=10
hrf=gloverhrf(25,1/upsample_factor)

fmat=np.zeros([nx,ny,nt*upsample_factor])
for i in range(ne):
    for j in range(int(ons[i,1]*upsample_factor)):
        fmat[:,:,int(ons[i,0]*upsample_factor)+j]=np.squeeze(dset[:,:,i])
fmat_conv=np.zeros([nx,ny,nt*upsample_factor+len(hrf)-1])
fmat_conv_intp=np.zeros([nx,ny,int(np.ceil(nt/TR))])

realtime=np.arange(0,nt,1/upsample_factor)
scantime=np.arange(0,nt,TR)
for i in range(nx):
    for j in range(ny):
        if np.sum(fmat[i,j,:])>0:
            fmat_conv[i,j,:]=np.convolve(fmat[i,j,:], hrf)
            fmat_conv[i,j,:]/=np.max(fmat_conv[i,j,:])
            fcon = interp1d(realtime,fmat_conv[i,j,0:int(nt*upsample_factor)],kind='cubic')
            fmat_conv_intp[i,j,:] = fcon(scantime)

filename = '/home/wouter/Fridge/Students/Glenn/subjects/fmatrix.h5'
f = h5py.File(filename, 'w')
f.create_dataset('fmatrix', data=fmat_conv_intp)
f.close
