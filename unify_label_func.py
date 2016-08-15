import os
import pdb
import glob as glob
import cPickle as pickle
import numpy as np
from scipy.io import loadmat,savemat
from DataPathclass import *
DataPathobj = DataPath(dataSource,VideoIndex)
from parameterClass import *
Parameterobj = parameter(dataSource,VideoIndex)

 
useCC = False
if Parameterobj.useWarpped:
    filePrefix = 'usewarpped_Aug15'
else:
    # filePrefix = 'Aug12'
    # filePrefix = 'Aug10'
    filePrefix = 'Aug15'

def unify_newLabel_to_existing(matfiles, LabelName, IDName):
    flab     = [] #final labels
    ftrjID   = [] #final trjID

    for matidx in range(len(matfiles)): 
        L1 = loadmat(matfiles[matidx])[LabelName][0]
        # L1 = L1+1 # class label starts from 1 instead of 0
        M1 = loadmat(matfiles[matidx])[IDName][0]

        if len(flab)>0:
            Labelnowmax = max(flab)
            L1 = L1+Labelnowmax+1
            commonidx = np.intersect1d(M1,ftrjID)  #trajectories existing in both 2 trucations

            print('flab : {0}, new labels : {1} ,common term : {2}').format(len(np.unique(flab)),len(np.unique(L1)),len(commonidx))
            for i in commonidx:
                labelnew = L1[M1==i][0]
                labelnow = np.array(flab)[ftrjID==i][0]
                idx1  = np.where(L1==labelnew)[0]
                L1[idx1] = labelnow  ## keep the first appearing label

        flab[:]   = flab +list(L1)
        ftrjID[:] = ftrjID + list(M1)
    
    ftrjID, indices= np.unique(ftrjID,return_index=True)
    flab = np.array(flab)[indices] 

    return flab, ftrjID


def unify_label(matfiles,savename,label_choice):
    DirName = ['upup','updown','downup','downdown']
    for dirii in range(len(DirName)):

        LabelName = label_choice+DirName[dirii]
        IDName = 'trjID_'+DirName[dirii]
        if useCC:
            LabelName = 'c_'+DirName[dirii]
            IDName = 'trjID_'+DirName[dirii]
        try:
            M1 = loadmat(matfiles[0])[IDName][0]
        except: ## if no this direction, key error, continue
            continue

        flab, ftrjID = unify_newLabel_to_existing(matfiles, LabelName, IDName)

        result          = {}
        result['label'] = flab
        result['trjID'] = ftrjID
        savename = os.path.join(DataPathobj.unifiedLabelpath,savename+label_choice)
        savemat(savename,result)


if __name__ == '__main__':
    label_choice = Parameterobj.clustering_choice
    if useCC:
        matfilesAll = sorted(glob.glob(DataPathobj.adjpath +filePrefix+'*.mat')) 
    else:
        matfilesAll = sorted(glob.glob(DataPathobj.sscpath +filePrefix+'*.mat'))

 
    numTrunc = len(matfilesAll)
    savename = ''
    if numTrunc<=200:
        if useCC:
            savename = 'concomp'+savename
        else:
            savename = 'Complete_result'+savename
        if Parameterobj.useWarpped:
            savename = 'usewarpped_'+savename

        unify_label(matfilesAll,savename,label_choice)
    else:
        for kk in range(0,numTrunc,25):
            print "saved trunk",str(kk+1).zfill(3),'to' ,str(min(kk+25,numTrunc)).zfill(3)
            matfiles = matfilesAll[kk:min(kk+25,numTrunc)]
            savename = os.path.join(DataPathobj.unifiedLabelpath,'result_'+label_choice+str(kk+1).zfill(3)+'-'+str(min(kk+25,numTrunc)).zfill(3))
            unify_label(matfiles,savename,label_choice)
            



