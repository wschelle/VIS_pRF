pro ro1prf_7TGEr

  rdir='/Fridge/users/wouter/Students/Glenn/subjects/'
  seq='GE'
  ses='ses-7T'+seq
  tasks=[seq+'_PRF_1',seq+'_PRF_2']
  ntask=n_elements(tasks)
  stdir=rdir+'task-prf.tsv'
  tgk=1.175
  
  subs=['sub-visual01','sub-visual02','sub-visual03','sub-visual05','sub-visual06',$
    'sub-visual08','sub-visual09','sub-visual10','sub-visual11','sub-visual12']
  nsubs=n_elements(subs)
  scandur=850.
  
  restore,rdir+'HRF_18700ms.sav'
;  hrf=reform(firz[-1,1,-1,*])
;  hrf-=hrf[0]
;  hrf/=max(hrf)

  restore,rdir+'prfstim28x28w.sav'
  sx=n_elements(prfstim[*,0,0])
  sy=n_elements(prfstim[0,*,0])
  
  restore,rdir+'fmatrix.sav'
  
  ii=0
  if file_test(rdir+subs[ii]+'/'+ses+'/func/'+tasks[0]+'/',/directory) then begin

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
  if not file_test(ddir,/directory) then mkdir, ddir
  if not file_test(pddir,/directory) then mkdir, pddir
  if not file_test(nddir,/directory) then mkdir, nddir
  if not file_test(sddir,/directory) then mkdir, sddir
 
  hrf=reform(firz[ii,1,-1,*])
  hrf-=hrf[0]
  hrf/=max(hrf)
  
  ;Read nifti
    x1=loadnii(nifti1,dims=dims,pixdim=pixdim)
    x2=loadnii(nifti2)
    dims=long(dims)

    fx=dims[1]
    fy=dims[2]
    fz=dims[3]
    nscans=dims[4]
    ;scandur=pixdim[4]*1000.
    ;if scandur ne 850. then print,'CHECK SCANDUR: NOT EQUAL TO 850ms'
    scandur=850.
    nvox=fx*fy*fz

    sm1=reform(x1,nvox,n_elements(x1[0,0,0,*]))
    sm2=reform(x2,nvox,n_elements(x2[0,0,0,*]))
    
    x1=0
    x2=0
    
    sm=transpose([transpose(sm1),transpose(sm2)])
    nscans=n_elements(sm[0,*])
    
    ;make simple mask
    mask1=intarr(nvox)
    mask1[where(mean(sm1,dimension=2) ge 100)]=1
    
    mask2=intarr(nvox)
    mask2[where(mean(sm2,dimension=2) ge 100)]=1
    
    mask=mask1*mask2
    
    sm1=0
    sm2=0
    mask1=0
    mask2=0

    ;*****Filtering of the data*****
    nrfilters = floor(2*(nscans*(scandur/1000.)*0.0125))
    filtermatrix=make_filter(nrfilters,nscans)
    taskreg=fltarr(nscans)
    taskreg[0:nscans/2-1]=1.
    filtermatrix=[filtermatrix,transpose(taskreg)]

    rp1=read_ascii(tdir1+task1+'-mp')
    rp1=rp1.(0)
    rp2=read_ascii(tdir2+task2+'-mp')
    rp2=rp2.(0)
    
    srp=size(rp1)
    rp1b=rp1*0
    for i=0,srp[1]-1 do for j=1,srp[2]-1 do rp1b[i,j]=rp1[i,j]-rp1[i,j-1]
    rp1c=fltarr(srp[1]*2,srp[2])
    for i=0,srp[1]-1 do rp1c[i,where(rp1b[i,*] gt 0)]=rp1b[i,where(rp1b[i,*] gt 0)]
    for i=0,srp[1]-1 do rp1c[i+srp[1],where(rp1b[i,*] lt 0)]=rp1b[i,where(rp1b[i,*] lt 0)]
    srp=size(rp2)
    rp2b=rp2*0
    for i=0,srp[1]-1 do for j=1,srp[2]-1 do rp2b[i,j]=rp2[i,j]-rp2[i,j-1]
    rp2c=fltarr(srp[1]*2,srp[2])
    for i=0,srp[1]-1 do rp2c[i,where(rp2b[i,*] gt 0)]=rp2b[i,where(rp2b[i,*] gt 0)]
    for i=0,srp[1]-1 do rp2c[i+srp[1],where(rp2b[i,*] lt 0)]=rp2b[i,where(rp2b[i,*] lt 0)]
    rp=transpose([transpose(rp1c),transpose(rp2c)])  
    grp=rp*0
    for i=0,11 do grp[i,*]=gauss_smooth(reform(rp[i,*]),tgk,/edge_mirror)
    rp=0

    for i=0L,nvox-1 do if mask[i] eq 1 then begin
      b=regress(filtermatrix,reform(sm[i,*]),yfit=yfit,const=const,/double)
      sm[i,*]=(sm[i,*]-yfit)/const*100.
    end
    filtermatrix=0

    smg=sm*0
    for i=0L,nvox-1 do if mask[i] eq 1 then smg[i,*]=gauss_smooth(reform(sm[i,*]),tgk,/edge_mirror)
    sm=0
    ;smg=sm
    
    for i=0L,nvox-1 do if mask[i] eq 1 then begin
      b=regress(grp,reform(smg[i,*]),yfit=yfit,const=const,/double)
      smg[i,*]-=yfit
      ;smg[i,*]+=const
    end
    grp=0
    
    if not file_test(sddir+sub+'-NORDIC_'+ses+'-prf-cmat.sav') then begin
      cmat=fltarr(nvox,sx,sy)
      for i=0L,nvox-1 do if mask[i] eq 1 then begin
        for j=0,sx-1 do for k=0,sy-1 do if stddev(fmatrix[j,k,*]) ne 0 then cmat[i,j,k]=correlate(reform(fmatrix[j,k,*]),reform(smg[i,*]))
      endif
      save,cmat,filename=sddir+sub+'-NORDIC_'+ses+'-prf-cmat.sav'
    endif else begin
      restore,sddir+sub+'-NORDIC_'+ses+'-prf-cmat.sav'
    endelse
    
    cmats=cmat*0
    for i=0L,nvox-1 do cmats[i,*,*]=smooth(reform(cmat[i,*,*]),6,/nan,/edge_zero)
    for j=0,sx-1 do for k=0,sy-1 do if stddev(fmatrix[j,k,*]) eq 0 then cmats[*,j,k]=0
    
    fmatrix=reform(fmatrix[*,*,0:nscans-1])

    gest=fltarr(nvox,6)
    cmat2=fltarr(nvox)
    ;fwhm=2*sqrt(2*alog(2))
    halfxy=(sx-1)/2.
    for i=0L,nvox-1 do if mask[i] eq 1 and max(cmats[i,*,*]) gt 0 then begin
      topfac=array_indices(reform(cmats[i,*,*]),where(cmats[i,*,*] eq max(cmats[i,*,*])))
      if n_elements(topfac) eq 2 then gest[i,2:3]=topfac else gest[i,2:3]=topfac[*,0]
      gest[i,1]=regress(reform(fmatrix[gest[i,2],gest[i,3],*]),reform(smg[i,*]),const=const)
      gest[i,0]=const
      tmpecc=sqrt((gest[i,2]-halfxy)^2. + (gest[i,3]-halfxy)^2.)
      gest[i,4]=tmpecc/3.
      gest[i,5]=1.
      cmat2[i]=cmat[i,gest[i,2],gest[i,3]]
    endif
    gest[where(reform(gest[*,4]) gt 0 and reform(gest[*,4]) lt 0.1),4]=0.1
    tmat=rval2tval(cmat2,nscans-1)
    cmat=0
    cmats=0

    npar=6.
    parinfo = replicate({fixed:0, limited:[0,0], limits:[0.,0.], step:0.}, npar)
    
    parinfo[5].fixed=1
    
    parinfo[1].limited(0)=1
    parinfo[1].limits(0)=0

    parinfo[2:4].limited(*)=1
    parinfo[2].limits(0)=-0.5
    parinfo[2].limits(1)=sx-0.5

    parinfo[3].limits(0)=-0.5
    parinfo[3].limits(1)=sy-0.5

    parinfo[4].limits(0)=0.1
    parinfo[4].limits(1)=sx*2
    
    parinfo[2:4].step=0.5

    err=fltarr(nscans)+1
    yfitz=fltarr(nvox,nscans)
    zfitz=fltarr(nvox,npar+1)
    count=0L
    errornodes=[]
    tc=fdr_t(tmat,nscans-1,mask=mask)
    totcount=n_elements(where(tmat gt tc))
    smg=float(smg)

  for i=0L,nvox-1 do if tmat[i] gt tc then begin
    p0=reform(gest[i,*])
    zfit=mpfitfun('prfratio2d',fmatrix,reform(smg[i,*]),err,p0,parinfo=parinfo,yfit=yfit,bestnorm=chisq,status=status,/quiet)
    if status gt 0 then begin
      zfitz[i,*]=[zfit,chisq]
      yfitz[i,*,*]=yfit
    endif else begin
      errornodes=[errornodes,i]
      print,'ERROR @ '+strcompress(i,/remove_all)
    endelse
    count++
    print,100./totcount*count
  end
  
    print,'# errornodes = '+strcompress(n_elements(errornodes),/remove_all)
    ;save,errornodes,filename=sddir+sub+'-NORDIC_'+ses+'-errornodes.sav'
    
    dfn=(npar-1)-1
    dfd=nscans-(npar-1)-1

    fval=fltarr(nvox)
    for i=0L,nvox-1 do if zfitz[i,-1] gt 0 then fval[i]=(total((mean(reform(smg[i,*]))-yfitz[i,*])^2.)/dfn)/(zfitz[i,-1]/dfd)
    r2val=fltarr(nvox)
    for i=0L,nvox-1 do if zfitz[i,-1] gt 0 then r2val[i]=correlate(reform(yfitz[i,*]),reform(smg[i,*]),/double)^2d

    zfitz=transpose([transpose(zfitz),transpose(fval)])
    zfitz=transpose([transpose(zfitz),transpose(r2val)])
    save,zfitz,filename=sddir+sub+'-NORDIC_'+ses+'-circprf-shrf-zfitz.sav'

    degvis=(12./sx)
    zfitz[*,4]*=degvis

    center=reform(zfitz[*,2:3])
    center[where(zfitz[*,7] gt 0),0]-=(sx-1)/2.
    center[where(zfitz[*,7] gt 0),1]-=(sy-1)/2.
    center*=degvis
    ecc=fltarr(nvox)
    for i=0L,nvox-1 do if fval[i] gt 0 then ecc[i]=sqrt(center[i,0]^2.+center[i,1]^2)
    pol=fltarr(nvox)
    for i=0L,nvox-1 do if fval[i] gt 0 then begin
      if center[i,1] ge 0 and ecc[i] ne 0 then pol[i]=acos(center[i,0]/ecc[i])
      if center[i,1] lt 0 then pol[i]= -acos(center[i,0]/ecc[i])
      pol[i]*=(180/!PI)
    endif
    pol[where(fval gt 0)]+=270.
    pol[where(fval gt 0 and pol ge 360)]-=360
    prfsize=reform(zfitz[*,4])

    pol2=reform(pol,fx,fy,fz)
    ecc2=reform(ecc,fx,fy,fz)
    prf2=reform(prfsize,fx,fy,fz)
    r2val2=reform(r2val,fx,fy,fz)
    amp2=reform(zfitz[*,1],fx,fy,fz)
    fval2=reform(fval,fx,fy,fz)

    hdr=readhdr(tdir1+'lay.nii')
    hdr.datatype=16
    hdr.bitpix=32
    wniihdr,hdr,pol2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-pol.nii'
    wniihdr,hdr,ecc2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-ecc.nii'
    wniihdr,hdr,prf2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-prf.nii'
    wniihdr,hdr,r2val2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-rsq.nii'
    wniihdr,hdr,amp2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-beta.nii'
    wniihdr,hdr,fval2,nddir+sub+'-NORDIC_'+ses+'-circprf-shrf-fval.nii'

    varea=loadnii(tdir1+'varea.nii')
    lay=loadnii(tdir1+'lay.nii')
    lay2=reform(lay,nvox)
    lay2[where(lay2 eq 1)]++
    lay2[where(lay2 eq 20)]--
    lay2[where(lay2 gt 0)]--
    lay2[where(lay2 gt 0)]/=6.
    lay2=ceil(lay2)
    varea=reform(varea,nvox)

    rth=0.05
    ct=colortable(25,ncolors=3)

    w=window(dimensions=[1800,600])
    for i=0,2 do if where(varea eq i+1 and r2val gt rth,/null) ne !NULL then begin $
      & p=plot(ecc[where(varea eq i+1 and lay2 eq 1 and r2val gt rth)],prfsize[where(varea eq i+1 and lay2 eq 1 and r2val gt rth)],$
        linestyle=6,symbol='o',sym_size=2.,sym_thick=3,layout=[3,1,i+1],/current,sym_color=reform(ct[i,*])*0.5,axis_style=1,xrange=[0,8],yrange=[0,8]) $
      & p=plot(ecc[where(varea eq i+1 and lay2 eq 2 and r2val gt rth)],prfsize[where(varea eq i+1 and lay2 eq 2 and r2val gt rth)],$
        linestyle=6,symbol='o',sym_size=2.,sym_thick=3,sym_color=reform(ct[i,*])*0.75,/overplot) $
      & p=plot(ecc[where(varea eq i+1 and lay2 eq 3 and r2val gt rth)],prfsize[where(varea eq i+1 and lay2 eq 3 and r2val gt rth)],$
        linestyle=6,symbol='o',sym_size=2.,sym_thick=3,sym_color=reform(ct[i,*]),/overplot) $
    & end
    w.save,pddir+sub+'-NORDIC_'+ses+'-prf-ecc_vs_prf-circ-shrf.png'
    w.close

  endif
end
